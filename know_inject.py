import torch
import torch.nn as nn
import math
from know_extra import KnowExtractor
from util import get_regularizer
class Identity():
    def __init__(self):
        pass
    def __call__(self, x):
        return x

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_memory=False
    ):

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        #
        # if self.is_decoder:
        #     outputs = outputs + (past_key_value,)
        return context_layer

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, NormFunc):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.NormFunc = NormFunc
        # self.dropout = nn.Dropout(hidden_dropout_prob)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.NormFunc(hidden_states)
        # hidden_states = self.sigmoid(hidden_states)
        hidden_states = hidden_states * input_tensor
        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob, NormFunc):
        super().__init__()
        self.self = MultiheadAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob, NormFunc)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs, hidden_states)
        # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return attention_output

class KnowInjector(nn.Module):
    def __init__(self, args=None):
        super(KnowInjector, self).__init__()

        self.args = args
        if args.geo in ['beta', 'beta_temp']:
            self.hidden_dim = self.args.hidden_dim * 2
            self.regularizer = Regularizer(1, 0.05, 1e9)
            # self.regularizer = Identity()
            # self.regularizer = nn.LayerNorm(self.hidden_size, eps=1e-12)
        elif args.geo in ['box', 'box_temp']:
            self.hidden_dim = self.args.hidden_dim * 2
            self.regularizer = Identity()
            # self.regularizer = nn.LayerNorm(self.hidden_size, eps=1e-12)
        elif args.geo in ['vec', 'vec_temp']:
            self.hidden_dim = self.args.hidden_dim
            self.regularizer = Identity()
            # self.regularizer = nn.LayerNorm(self.hidden_size, eps=1e-12)
        elif args.geo == 'cone':
            self.hidden_dim = self.args.hidden_dim * 2
            self.regularizer = Identity()
            # self.regularizer = nn.LayerNorm(self.hidden_size, eps=1e-12)
        elif args.geo == 'fuzzy':
            self.hidden_dim = self.args.hidden_dim
            self.regularizer = Identity()
        elif args.geo == 'mlp2vec':
            self.hidden_dim = self.args.hidden_dim
            self.regularizer = Identity()
        self.ke = KnowExtractor(args=self.args)
        self.output = SelfOutput(hidden_size=self.hidden_dim, hidden_dropout_prob=0.1, NormFunc=self.regularizer)
        # self.encoder_layer = TransformerLayer(hidden_size=self.hidden_dim,
        #                                       num_attention_heads=16,
        #                                       attention_probs_dropout_prob=0.1,
        #                                       hidden_dropout_prob=0.1,
        #                                       NormFunc=self.regularizer)

    def forward(self, input_pattern, input_embeddings, text):


        if input_pattern != 'negative':
            input_embeddings = input_embeddings.unsqueeze(1)
        else:
            batch_size, negative_size, _ = input_embeddings.size()
            input_embeddings = input_embeddings.view(batch_size * negative_size, -1).unsqueeze(1)

        extractor_output = self.ke(keg_query_embeddings=input_embeddings, text=text)

        know_inject_embeddings = self.output(hidden_states=extractor_output, input_tensor=input_embeddings).squeeze(1)


        # know_inject_embeddings = self.encoder_layer(hidden_states=input_embeddings,
        #                                             encoder_hidden_states=extractor_output).squeeze(1)

        return know_inject_embeddings
