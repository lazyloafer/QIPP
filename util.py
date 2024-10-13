import numpy as np
import random
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from typing import Optional

pi = 3.14159265358979323846

query_name_dict = {
    ('e',('r',)): '1p',
    ('e', ('r', 'r')): '2p',
    ('e', ('r', 'r', 'r')): '3p',
    (('e', ('r',)), ('e', ('r',))): '2i',
    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
}
query_structure_list = list(query_name_dict.keys())  # query_structure_list[0] -> query_structure of index 0
query_structure2idx = {s: i for i, s in enumerate(query_structure_list)}  # {('e',('r',)):0}

def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return

def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries

def Identity(x):
    return x

class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate

class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding

class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        layer1_act = F.relu(self.layer1(all_embeddings))  # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, batch_size, dim)

        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=0)

        return alpha_embedding, beta_embedding

class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers, with_regular=True):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer
        self.with_regular = with_regular

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)
        if self.with_regular:
            x = self.projection_regularizer(x)

        return x

class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, r_embedding_arg):
        x = torch.cat([source_embedding_axis + r_embedding_axis, source_embedding_arg + r_embedding_arg], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg = torch.chunk(x, 2, dim=-1)
        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        return axis_embeddings, arg_embeddings

class ConeIntersection(nn.Module):
    def __init__(self, dim, drop):
        super(ConeIntersection, self).__init__()
        self.dim = dim
        self.layer_axis1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_arg1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_axis2 = nn.Linear(self.dim, self.dim)
        self.layer_arg2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_axis1.weight)
        nn.init.xavier_uniform_(self.layer_arg1.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, axis_embeddings, arg_embeddings):
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings], dim=-1)
        axis_layer1_act = F.relu(self.layer_axis1(logits))

        axis_attention = F.softmax(self.layer_axis2(axis_layer1_act), dim=0)

        x_embeddings = torch.cos(axis_embeddings)
        y_embeddings = torch.sin(axis_embeddings)
        x_embeddings = torch.sum(axis_attention * x_embeddings, dim=0)
        y_embeddings = torch.sum(axis_attention * y_embeddings, dim=0)

        # when x_embeddings are very closed to zero, the tangent may be nan
        # no need to consider the sign of x_embeddings
        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] = axis_embeddings[indicator_two] + pi
        axis_embeddings[indicator_three] = axis_embeddings[indicator_three] - pi

        # DeepSets
        arg_layer1_act = F.relu(self.layer_arg1(logits))
        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings = self.drop(arg_embeddings)
        arg_embeddings, _ = torch.min(arg_embeddings, dim=0)
        arg_embeddings = arg_embeddings * gate

        return axis_embeddings, arg_embeddings

class ConeNegation(nn.Module):
    def __init__(self):
        super(ConeNegation, self).__init__()

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - arg_embedding

        return axis_embedding, arg_embedding

class FuzzProjection(nn.Module):
    # def __init__(self, entity_dim, logic_type, regularizer_setting):
    def __init__(
            self,
            nrelation,
            entity_dim,
            logic_type,
            regularizer_setting,
            relation_dim,
            projection_dim,
            num_layers,
            projection_type,
            num_rel_base,  # for 'rtransform'
    ):
        super(FuzzProjection, self).__init__()
        self.logic = logic_type

        # # temporary testing
        # regularizer_setting = {
        #         'type': 'sigmoid',
        #     }

        self.regularizer = get_regularizer(regularizer_setting, entity_dim, neg_input_possible=True)
        # for projection
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.projection_type = projection_type

        self.dual = regularizer_setting['dual']

        # mlp
        if projection_type == 'mlp':
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))  # same dim
            nn.init.uniform_(tensor=self.relation_embedding, a=0, b=1)

            # mlp
            self.hidden_dim = projection_dim
            self.num_layers = num_layers
            self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  # 1st layer
            self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)  # final layer
            for nl in range(2, num_layers + 1):
                setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
            for nl in range(num_layers + 1):
                nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        elif projection_type == 'rtransform':
            n_base = num_rel_base
            if not self.dual:
                self.hidden_dim = entity_dim
                self.rel_base = nn.Parameter(torch.zeros(n_base, self.hidden_dim, self.hidden_dim))
                # nn.init.uniform_(self.rel_base, a=0, b=1e-2)
                self.rel_bias = nn.Parameter(torch.zeros(n_base, self.hidden_dim))
                self.rel_att = nn.Parameter(torch.zeros(nrelation, n_base))
                self.norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

                # new initialization
                torch.nn.init.orthogonal_(self.rel_base)
                torch.nn.init.xavier_normal_(self.rel_bias)
                torch.nn.init.xavier_normal_(self.rel_att)

            else:
                self.hidden_dim = entity_dim // 2

                # for property vals
                self.rel_base1 = nn.Parameter(torch.randn(n_base, self.hidden_dim, self.hidden_dim))
                nn.init.uniform_(self.rel_base1, a=0, b=1e-2)
                self.rel_bias1 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
                self.rel_att1 = nn.Parameter(torch.randn(nrelation, n_base))
                self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

                # for property weights
                self.rel_base2 = nn.Parameter(torch.randn(n_base, self.hidden_dim, self.hidden_dim))
                nn.init.uniform_(self.rel_base2, a=0, b=1e-2)
                self.rel_bias2 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
                self.rel_att2 = nn.Parameter(torch.randn(nrelation, n_base))
                self.norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        elif projection_type == 'transe':
            self.hidden_dim = entity_dim
            self.rel_trans = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
            self.rel_bias = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
            torch.nn.init.xavier_normal_(self.rel_trans)
            torch.nn.init.xavier_normal_(self.rel_bias)
            self.norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

    def forward(self, e_embedding, rid):
        if self.projection_type == 'mlp':
            r_embedding = torch.index_select(self.relation_embedding, dim=0, index=rid)
            x = torch.cat([e_embedding, r_embedding], dim=-1)
            for nl in range(1, self.num_layers + 1):
                x = F.relu(getattr(self, "layer{}".format(nl))(x))
            x = self.layer0(x)
            x = self.regularizer(x)
            return x

        if self.projection_type == 'rtransform':
            if not self.dual:
                project_r = torch.einsum('br,rio->bio', self.rel_att[rid], self.rel_base)
                if self.rel_bias.shape[0] == self.rel_base.shape[0]:
                    bias = torch.einsum('br,ri->bi', self.rel_att[rid], self.rel_bias)
                else:
                    bias = self.rel_bias[rid]
                output = torch.einsum('bio,bi->bo', project_r, e_embedding) + bias
                output = self.norm(output)
            else:
                e_embedding1, e_embedding2 = torch.chunk(e_embedding, 2, dim=-1)
                project_r1 = torch.einsum('br,rio->bio', self.rel_att1[rid], self.rel_base1)
                bias1 = self.rel_bias1[rid]
                output1 = torch.einsum('bio,bi->bo', project_r1, e_embedding1) + bias1
                output1 = self.norm1(output1)

                project_r2 = torch.einsum('br,rio->bio', self.rel_att2[rid], self.rel_base2)
                bias2 = self.rel_bias2[rid]
                output2 = torch.einsum('bio,bi->bo', project_r2, e_embedding2) + bias2
                output2 = self.norm1(output2)

                output = torch.cat((output1, output2), dim=-1)

            output = self.regularizer(output)
            return output

        if self.projection_type == 'transe':
            r_trans = torch.index_select(self.rel_trans, dim=0, index=rid)
            r_bias = torch.index_select(self.rel_bias, dim=0, index=rid)
            output = e_embedding * r_trans + r_bias

            output = self.norm(output)
            output = self.regularizer(output)
            return output

class FuzzConjunction(nn.Module):
    def __init__(self, entity_dim, logic_type, regularizer_setting, use_attention='False', godel_gumbel_beta=0.01):
        super(FuzzConjunction, self).__init__()
        self.logic = logic_type
        self.regularizer = get_regularizer(regularizer_setting, entity_dim)
        self.use_attention = use_attention
        self.entity_dim = entity_dim

        if logic_type == 'godel_gumbel':
            self.godel_gumbel_beta = godel_gumbel_beta
        if use_attention:
            self.conjunction_layer1 = nn.Linear(self.entity_dim, self.entity_dim)
            # self.conjunction_layer2 = nn.Linear(self.entity_dim, self.entity_dim)
            self.conjunction_layer2 = nn.Linear(self.entity_dim, 1)  # no dimension-wise attention
            nn.init.xavier_uniform_(self.conjunction_layer1.weight)
            nn.init.xavier_uniform_(self.conjunction_layer2.weight)
        self.norm = nn.LayerNorm(entity_dim, elementwise_affine=False)

    def forward(self, embeddings):
        """
        :param embeddings: shape (# of sets, batch, dim).
        :return embeddings: shape (batch, dim)
        """
        if self.logic == 'godel':
            if self.logic == 'godel':
                # conjunction(x,y) = min{x,y}
                embeddings, _ = torch.min(embeddings, dim=0)
            elif self.logic == 'godel_gumbel':
                # soft way to compute min
                embeddings = -self.godel_gumbel_beta * torch.logsumexp(
                    -embeddings / self.godel_gumbel_beta,
                    0
                )
            return embeddings
        else:  # logic == product
            if self.logic == 'luka':
                # conjunction(x,y) = max{0, x+y-1}
                embeddings = torch.sum(embeddings, dim=0) - embeddings.shape[0] + 1
            elif self.logic == 'product':
                if not self.use_attention:
                    # conjunction(x,y) = xy
                    embeddings = torch.prod(embeddings, dim=0)
                else:
                    attention = self.get_conjunction_attention(embeddings)
                    # attention conjunction(x,y) = (x^p)*(y^q), p+q=1
                    # compute in log scale
                    epsilon = 1e-7  # avoid torch.log(0)
                    embeddings = torch.log(embeddings + epsilon)
                    embeddings = torch.exp(torch.sum(embeddings * attention, dim=0))
            embeddings = self.norm(embeddings)
            return self.regularizer(embeddings)

    def get_conjunction_attention(self, embeddings):
        layer1_act = F.relu(self.conjunction_layer1(embeddings))  # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.conjunction_layer2(layer1_act) / torch.sqrt(self.entity_dim),
                              dim=0)  # (num_conj, batch_size, 1)
        return attention

class FuzzDisjunction(nn.Module):
    def __init__(self, entity_dim, logic_type, regularizer_setting, godel_gumbel_beta=0.01):
        super(FuzzDisjunction, self).__init__()
        self.logic = logic_type
        self.regularizer = get_regularizer(regularizer_setting, entity_dim)

        if logic_type == 'godel_gumbel':
            self.godel_gumbel_beta = godel_gumbel_beta

        self.norm = nn.LayerNorm(entity_dim, elementwise_affine=False)

    def forward(self, embeddings):
        """
        :param embeddings: shape (# of sets, batch, dim).
        :return embeddings: shape (batch, dim)
        """
        if self.logic == 'godel':
            if self.logic == 'godel':
                # disjunction(x,y) = max{x,y}
                embeddings, _ = torch.max(embeddings, dim=0)
                return embeddings
            elif self.logic == 'godel_gumbel':
                # soft way to compute max
                embeddings = self.godel_gumbel_beta * torch.logsumexp(
                    embeddings / self.godel_gumbel_beta,
                    0
                )
            return embeddings
        else:
            if self.logic == 'luka':
                # disjunction(x,y) = min{1, x+y}
                embeddings = torch.sum(embeddings, dim=0)
            else:  # self.logic == 'product'
                # disjunction(x,y) = x+y-xy
                embeddings = torch.sum(embeddings, dim=0) - torch.prod(embeddings, dim=0)
            return self.regularizer(embeddings)

class FuzzNegation(nn.Module):
    def __init__(self, entity_dim, logic_type, regularizer_setting):
        super(FuzzNegation, self).__init__()
        self.logic = logic_type
        self.regularizer = get_regularizer(regularizer_setting, entity_dim)

    def forward(self, embeddings):
        """
        :param embeddings: shape (# of sets, batch, dim).
        :return embeddings: shape (batch, dim)
        """
        # negation(x) = 1-x
        return 1 - embeddings

class ProjectionMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(ProjectionMLP, self).__init__()
        self.n_layers = n_layers
        for i in range(1, self.n_layers+1):
            setattr(self, "proj_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers+1):
            x = F.relu(getattr(self, "proj_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x

class AndMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(AndMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "and_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "and_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x

class OrMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(OrMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "or_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "or_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x

class NotMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(NotMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "not_layer_{}".format(i), nn.Linear(entity_dim, entity_dim))
        self.last_layer = nn.Linear(entity_dim, entity_dim)

    def forward(self, x):
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "not_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x

class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class SigmoidRegularizer(nn.Module):
    def __init__(self, vector_dim, dual=False):
        """
        :param dual: Split each embedding into 2 chunks.
                     The first chunk is property values and the second is property weight.
                     Do NOT sigmoid the second chunk.
        """
        super(SigmoidRegularizer, self).__init__()
        self.vector_dim = vector_dim
        # initialize weight as 8 and bias as -4, so that 0~1 input still mostly falls in 0~1
        self.weight = nn.Parameter(torch.Tensor([8]))
        self.bias = nn.Parameter(torch.Tensor([-4]))

        self.dual = dual

    def __call__(self, entity_embedding):
        if not self.dual:
            return torch.sigmoid(entity_embedding * self.weight + self.bias)
        else:
            # The first half is property values and the second is property weight.
            # Do NOT sigmoid the second chunk. The second chunk will be free parameters
            entity_vals, entity_val_weights = torch.chunk(entity_embedding, 2, dim=-1)
            entity_vals = torch.sigmoid(entity_vals * self.weight + self.bias)
            return torch.cat((entity_vals, entity_val_weights), dim=-1)

    def soft_discretize(self, entity_embedding, temperature=10):
        return torch.sigmoid((entity_embedding * self.weight + self.bias) * temperature)  # soft

    def hard_discretize(self, entity_embedding, temperature=10, thres=0.5):
        discrete = self.soft_discretize(entity_embedding, temperature)
        discrete[discrete >= thres] = 1
        discrete[discrete < thres] = 0
        return discrete

class MatrixSoftmaxRegularizer(nn.Module):
    def __init__(self, vector_dim, k):
        """
        :param k: split the vector into matrix, k elements per row. k has to be a factor of vector_dim
        """
        super(MatrixSoftmaxRegularizer, self).__init__()
        self.vector_dim = vector_dim
        self.k = k
        self.softmax = nn.Softmax(dim=-1)
        self.softmax_weight = nn.Parameter(torch.full((self.vector_dim,), fill_value=0.1))
        self.softmax_bias = nn.Parameter(torch.Tensor([0]))

    def __call__(self, entity_embedding):
        """
        :param entity_embedding: shape [batch_size, dim]
        """
        transformed = entity_embedding * self.softmax_weight + self.softmax_bias

        # reshape the last dimension into a matrix
        dims, last_dim = entity_embedding.size()[:-1], entity_embedding.size()[-1]
        n_row = last_dim // self.k
        n_col = self.k

        reshaped = transformed.view(*dims, n_row, n_col)
        reshaped = self.softmax(reshaped)  # softmax along the last dimension
        return reshaped.view(*dims, last_dim)  # change to original shape

    def reshape_to_matrix(self, entity_embedding):
        # reshape the last dimension into a matrix
        dims, last_dim = entity_embedding.size()[:-1], entity_embedding.size()[-1]
        n_row = last_dim // self.k
        n_col = self.k

        reshaped = entity_embedding.view(*dims, n_row, n_col)
        return reshaped

    def reshape_to_vector(self, entity_embedding_matrix):
        dims, n_row, n_col = entity_embedding_matrix.size()[:-2], entity_embedding_matrix.size()[-1], \
                             entity_embedding_matrix.size()[-2]
        last_dim = n_row * n_col
        return entity_embedding_matrix.view(*dims, last_dim)

class VectorSoftmaxRegularizer(nn.Module):
    def __init__(self, vector_dim):
        """
        :param k: split the vector into matrix, k elements per row. k has to be a factor of vector_dim
        """
        super(VectorSoftmaxRegularizer, self).__init__()
        self.vector_dim = vector_dim
        self.softmax = nn.Softmax(dim=-1)
        # self.softmax_weight = nn.Parameter(torch.Tensor([1]))
        # self.softmax_bias = nn.Parameter(torch.Tensor([0]))

        # a fixed small weight is much better than learnable weight
        self.softmax_weight = 0.01
        self.softmax_bias = 0

    def __call__(self, entity_embedding):
        """
        :param entity_embedding: shape [batch_size, dim]
        """
        # softmax along the last dimension
        entity_embedding = entity_embedding * self.softmax_weight + self.softmax_bias
        return self.softmax(entity_embedding)

class VectorSigmoidSumRegularizer(nn.Module):
    def __init__(self, vector_dim, neg_input_possible=False, use_layernorm=False):
        super(VectorSigmoidSumRegularizer, self).__init__()

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            # applies layernorm before sigmoid
            # initialize weight as 8 and bias as -4, so that 0~1 input still mostly falls in 0~1
            self.weight = nn.Parameter(torch.Tensor([5]), requires_grad=True)
            self.bias = nn.Parameter(torch.Tensor([0]), requires_grad=True)

            self.layernorm = nn.LayerNorm(vector_dim, elementwise_affine=False)

        else:
            # initialize weight as 8 and bias as -4, so that 0~1 input still mostly falls in 0~1
            self.weight = nn.Parameter(torch.Tensor([8]), requires_grad=False)
            self.bias = nn.Parameter(torch.Tensor([-4]), requires_grad=False)

    def forward(self, embeddings):
        """
        :param embeddings: shape [batch_size, dim]
        """
        x = embeddings
        if self.use_layernorm:
            x = self.layernorm(x)
        x = torch.sigmoid(x * self.weight + self.bias)  # shift to non-negative
        x = F.normalize(x, p=1, dim=-1)  # L1 normalize along the last dimension
        return x

class MatrixSumRegularizer(nn.Module):
    """
    For set representation (computation graph node representation)
    """

    def __init__(self, vector_dim, k, neg_input_possible=False):
        """
        :param k: split the vector into matrix, k elements per row. k has to be a factor of vector_dim
        """
        super(MatrixSumRegularizer, self).__init__()
        self.vector_dim = vector_dim
        self.k = k
        self.neg_input_possible = neg_input_possible  # True for entity regularizer, False for set regularizer

    def forward(self, embeddings):
        """
        :param embeddings: shape [batch_size, dim]
        """
        # reshape the last dimension into a matrix
        dims, last_dim = embeddings.size()[:-1], embeddings.size()[-1]
        n_row = last_dim // self.k
        n_col = self.k

        reshaped = embeddings.view(*dims, n_row, n_col)

        if self.neg_input_possible:
            # shift to non-negative
            reshaped = torch.relu(reshaped)

            # min_per_row, _ = torch.min(reshaped, dim=-1,keepdim=True)
            # min_per_row[min_per_row>=0] = 0  # if min_per_row is positive, no need to shift
            # reshaped -= min_per_row  # shift by the minimum negative value

        # L1 normalize
        reshaped = F.normalize(reshaped, p=1, dim=-1)  # L1 normalize along the last dimension
        reshaped = reshaped.view(*dims, last_dim)  # change to original shape
        return reshaped

    def reshape_to_matrix(self, embeddings):
        # reshape the last dimension into a matrix
        dims, last_dim = embeddings.size()[:-1], embeddings.size()[-1]
        n_row = last_dim // self.k
        n_col = self.k

        reshaped = embeddings.view(*dims, n_row, n_col)
        return reshaped

    def reshape_to_vector(self, embeddings_matrix):
        dims, n_row, n_col = embeddings_matrix.size()[:-2], embeddings_matrix.size()[-2], embeddings_matrix.size()[-1]
        last_dim = n_row * n_col
        return embeddings_matrix.view(*dims, last_dim)

    def hard_discretize(self, embeddings):
        """
        Discretize as a matrix. k entries per row => one '1' per row.
        No gradient.
        No normalization added. (not needed)
        :param embeddings: shape [batch_size, 1 or num_neg, entity_dim], 0<=embeddings[i]<=1
        :return y_hard: [batch_size, 1 or num_neg, entity_dim]
        """
        y = self.reshape_to_matrix(embeddings)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)  # shape [*dims, entity_dim//k, k]
        y_hard = self.reshape_to_vector(y_hard)
        return y_hard

    def soft_discretize(self, embeddings, gumbel_temperature):
        """
        Discretize as a matrix. k entries per row => one '1' per row.
        Soft discretize using Gumbel softmax.
        No normalization added. (not needed)
        :param embeddings: shape [batch_size, 1 or num_neg, entity_dim], 0<=embeddings[i]<=1
        :param gumbel_temperature: max(0.5, exp(-rt)), r={1e-4, 1e-5}
        :return y_hard: [batch_size, 1 or num_neg, entity_dim]
        """
        y = self.reshape_to_matrix(embeddings)
        eps = 1e-5
        log_y = torch.log(y + eps)
        y_soft = F.gumbel_softmax(log_y, tau=gumbel_temperature, hard=False)
        y_soft = self.reshape_to_vector(y_soft)
        return y_soft

    def L1_normalize(self, embeddings):
        """
        :param embeddings: shape [batch_size, dim]
        :return: shape [batch_size, dim]
        """
        k = self.k
        # reshape the last dimension into a matrix
        dims, last_dim = embeddings.size()[:-1], embeddings.size()[-1]
        n_row = last_dim // k
        n_col = k

        reshaped = embeddings.view(*dims, n_row, n_col)

        # L1 normalize
        reshaped = F.normalize(reshaped, p=1, dim=-1)  # L1 normalize along the last dimension
        reshaped = reshaped.view(*dims, last_dim)  # change to original shape
        return reshaped

    def get_num_distributions(self):
        return self.vector_dim // self.k

class MatrixSigmoidSumRegularizer(MatrixSumRegularizer):
    def __init__(self, vector_dim, k, neg_input_possible=False):
        super(MatrixSigmoidSumRegularizer, self).__init__(vector_dim, k, neg_input_possible)
        # initialize weight as 8 and bias as -4, so that 0~1 input still mostly falls in 0~1
        self.weight = nn.Parameter(torch.Tensor([1]))
        self.bias = nn.Parameter(torch.Tensor([0]))

    def forward(self, embeddings):
        """
        :param embeddings: shape [batch_size, dim]
        """
        # reshape the last dimension into a matrix
        dims, last_dim = embeddings.size()[:-1], embeddings.size()[-1]
        n_row = last_dim // self.k
        n_col = self.k

        reshaped = embeddings.view(*dims, n_row, n_col)

        if self.neg_input_possible:  # for entity free parameters
            # shift to non-negative
            reshaped = torch.sigmoid(reshaped * self.weight + self.bias)

        # L1 normalize
        reshaped = F.normalize(reshaped, p=1, dim=-1)  # L1 normalize along the last dimension
        reshaped = reshaped.view(*dims, last_dim)  # change to original shape
        return reshaped

class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim, act, self_included, neighbor_ent_type_samples):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included
        self.neighbor_ent_type_samples = neighbor_ent_type_samples

    def forward(self, self_vectors, neighbor_vectors):
        outputs = self._call(self_vectors, neighbor_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        pass

class EntityTypeAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, act=lambda x: x, self_included=True, with_sigmoid=False, neighbor_ent_type_samples=32):
        super(EntityTypeAggregator, self).__init__(input_dim, output_dim, act, self_included, neighbor_ent_type_samples)
        self.proj_layer = HighwayNetwork(neighbor_ent_type_samples, 1, 2, activation=nn.Sigmoid())

        multiplier = 2 if self_included else 1
        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        self.with_sigmoid = with_sigmoid

    def _call(self, self_vectors, neighbor_vectors):
        neighbor_vectors = torch.transpose(neighbor_vectors, 1, 2)
        neighbor_vectors = self.proj_layer(neighbor_vectors)
        neighbor_vectors = torch.transpose(neighbor_vectors, 1, 2)
        neighbor_vectors = neighbor_vectors.squeeze(1)

        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])
            output = torch.cat([self_vectors, neighbor_vectors], dim=-1)
        output = self.layer(output)
        output = output.view([-1, self.output_dim])
        if self.with_sigmoid:
            output = torch.sigmoid(output)

        return self.act(output)

class HighwayNetwork(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               n_layers: int,
               activation: Optional[nn.Module] = None):
    super(HighwayNetwork, self).__init__()
    self.n_layers = n_layers
    self.nonlinear = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    self.gate = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    for layer in self.gate:
      layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
    self.final_linear_layer = nn.Linear(input_dim, output_dim)
    self.activation = nn.ReLU() if activation is None else activation
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    for layer_idx in range(self.n_layers):
      gate_values = self.sigmoid(self.gate[layer_idx](inputs))
      nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
      inputs = gate_values * nonlinear + (1. - gate_values) * inputs
    return self.final_linear_layer(inputs)

class Match(nn.Module):
    def __init__(self, hidden_size, with_sigmoid=False):
        super(Match, self).__init__()
        self.map_linear = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.trans_linear = nn.Linear(hidden_size, hidden_size)
        self.with_sigmoid = with_sigmoid

    def forward(self, inputs):
        proj_p, proj_q = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        att_norm = torch.nn.functional.softmax(att_weights, dim=-1)
        att_vec = att_norm.bmm(proj_q)
        elem_min = att_vec - proj_p
        elem_mul = att_vec * proj_p
        all_con = torch.cat([elem_min, elem_mul], 2)
        output = self.map_linear(all_con)
        if self.with_sigmoid:
            output = torch.sigmoid(output)
        return output

class RelationTypeAggregator(nn.Module):
    def __init__(self, hidden_size, with_sigmoid=False):
        super(RelationTypeAggregator, self).__init__()
        self.linear = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.with_sigmoid = with_sigmoid

    def forward(self, inputs):
        p, q = inputs
        lq = self.linear2(q)
        lp = self.linear2(p)
        mid = nn.Sigmoid()(lq+lp)
        output = p * mid + q * (1-mid)
        output = self.linear(output)
        if self.with_sigmoid:
            output = torch.sigmoid(output)
        return output

def get_regularizer(regularizer_setting, entity_dim, neg_input_possible=True, entity=False):
    """
    :param neg_input_possible: for matrix_L1 (class MatrixSumRegularizer)
    :param dual: only apply regularizer to the first half embeddings (after chunk dim=-1) (for sigmoid only)
    """
    if entity:
        key = 'e_reg_type'
    else:
        key = 'type'

    add_layernorm = regularizer_setting['e_layernorm']
    if regularizer_setting[key] == '01':
        regularizer = Regularizer(base_add=0, min_val=0, max_val=1)
    elif regularizer_setting[key] == 'matrix_softmax':
        prob_dim = regularizer_setting['prob_dim']
        regularizer = MatrixSoftmaxRegularizer(entity_dim, prob_dim)
    elif regularizer_setting[key] == 'vector_softmax':
        regularizer = VectorSoftmaxRegularizer(entity_dim)
    elif regularizer_setting[key] == 'sigmoid':
        regularizer = SigmoidRegularizer(entity_dim, dual=regularizer_setting['dual'])
    elif regularizer_setting[key] == 'matrix_L1':
        prob_dim = regularizer_setting['prob_dim']
        regularizer = MatrixSumRegularizer(entity_dim, prob_dim, neg_input_possible)
    elif regularizer_setting[key] == 'matrix_sigmoid_L1':
        prob_dim = regularizer_setting['prob_dim']
        regularizer = MatrixSigmoidSumRegularizer(entity_dim, prob_dim, neg_input_possible)
    elif regularizer_setting[key] == 'vector_sigmoid_L1':
        regularizer = VectorSigmoidSumRegularizer(entity_dim, neg_input_possible, add_layernorm)
    else:
        raise ValueError(
            'regularizer must in [01, matrix_softmax, vector_softmax, sigmoid, matrix_L1, matrix_sigmoid_L1, vector_sigmoid_L1]')
    return regularizer

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    # print(len(values), size)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y

def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y