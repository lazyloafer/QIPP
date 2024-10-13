import math
import torch
import torch.nn as nn
from transformers import AutoConfig
# from kbc.src.PLM.language_model import PLM
from PLM.language_model import PLM

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
    def __init__(self, in_hidden_size, out_hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        if out_hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({out_hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(out_hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.all_head_size, self.all_head_size)
        self.key = nn.Linear(in_hidden_size, self.all_head_size)
        self.value = nn.Linear(in_hidden_size, self.all_head_size)

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

class KnowExtractor(nn.Module):
    def __init__(self, args=None):
        super(KnowExtractor, self).__init__()

        self.args = args
        self.hidden_dim = self.args['rank']
        self.plm_config = AutoConfig.from_pretrained(args['plm_name'])
        self.plm = PLM(config=self.plm_config, args=self.args)

        # self.extractor_idx = torch.arange(0, self.args.extractor_size).int().cuda()
        # self.extractor_embeddings = nn.Embedding(args.extractor_size, self.plm_config.hidden_size)
        self.extractor_attention_layer = MultiheadAttention(in_hidden_size=self.plm_config.hidden_size,
                                                            out_hidden_size=self.hidden_dim,
                                                            num_attention_heads=self.args['extractor_num_attention_heads'],
                                                            attention_probs_dropout_prob=self.plm_config.attention_probs_dropout_prob)
        # self.extractor_linear = nn.Linear(self.hidden_dim, self.hidden_dim)

    def extractor(self, keg_query_embeddings, queries_text_output, attention_mask):
        # queries_text_output = self.regularizer(queries_text_output)
        attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        # extractor_input_embed = self.extractor_embeddings(
        #     self.extractor_idx.unsqueeze(0).expand(queries_text_output.size(0), -1)
        # )
        extractor_output = self.extractor_attention_layer(hidden_states=keg_query_embeddings,
                                                          encoder_hidden_states=queries_text_output,
                                                          encoder_attention_mask=attention_mask)
        # extractor_output = self.regularizer(self.extractor_linear(extractor_output))
        return extractor_output

    def forward(self, keg_query_embeddings, text):

        queries_text_output = self.plm(input_ids=text['input_ids'],
                                       attention_mask=text['attention_mask'])

        extractor_output = self.extractor(keg_query_embeddings=keg_query_embeddings,
                                          queries_text_output=queries_text_output,
                                          attention_mask=text['attention_mask'])

        return extractor_output