#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from tqdm import tqdm
import pickle
from transformers import AutoConfig, AutoTokenizer
from util import collate_tokens, pi, query_name_dict, query_structure_list, Identity, Regularizer, AngleScale, convert_to_axis, convert_to_arg, \
    BoxOffsetIntersection, CenterIntersection, BetaIntersection, BetaProjection, \
    ConeProjection, ConeIntersection, ConeNegation, \
    FuzzProjection, FuzzConjunction, FuzzDisjunction, FuzzNegation, get_regularizer, \
    ProjectionMLP, OrMLP, AndMLP, NotMLP,\
    EntityTypeAggregator, RelationTypeAggregator, Match
from know_inject import KnowInjector
import os

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None, args=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size,
                                                                               1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict
        self.args = args

        self.know_injector = KnowInjector(args=self.args)

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        if self.geo == 'box':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))  # centor for entities
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            activation, cen = box_mode
            self.cen = cen  # hyperparameter that balances the in-box distance and the out-box distance
            if activation == 'none':
                self.func = Identity
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus
        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))  # center for entities
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        elif self.geo == 'beta':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim * 2))  # alpha and beta
            self.entity_regularizer = Regularizer(1, 0.05,
                                                  1e9)  # make sure the parameters of beta embeddings are positive
            self.projection_regularizer = Regularizer(1, 0.05,
                                                      1e9)  # make sure the parameters of beta embeddings after relation projection are positive
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        elif self.geo == 'cone':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim),
                                                 requires_grad=True)  # axis for entities
            self.angle_scale = AngleScale(self.embedding_range.item())  # scale axis embeddings to [-pi, pi]

            self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)

            self.axis_scale = 1.0
            self.arg_scale = 1.0
            self.cone_center_reg = self.args.cone_center_reg

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding,
                a=0.,
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)
        elif self.geo == 'vec':
            self.center_net = CenterIntersection(self.entity_dim)
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2,
                                                 self.relation_dim,
                                                 hidden_dim,
                                                 self.projection_regularizer,
                                                 num_layers)
        elif self.geo == 'cone':
            self.axis_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.axis_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            self.arg_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.arg_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            self.cone_proj = ConeProjection(self.entity_dim, 1600, 2)
            self.cone_intersection = ConeIntersection(self.entity_dim, self.args.cone_drop)
            self.cone_negation = ConeNegation()


    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict,
                text=None, entity_embeddings=None):
        if self.geo == 'box':
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                    batch_idxs_dict, text, entity_embeddings)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                    batch_idxs_dict, text, entity_embeddings)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                     batch_idxs_dict, text, entity_embeddings)
        elif self.geo == 'cone':
            return self.forward_cone(positive_sample, negative_sample, subsampling_weight, batch_queries_dict,
                                     batch_idxs_dict, text, entity_embeddings)

    def embed_query_box(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                if self.use_cuda:
                    offset_embedding = torch.zeros_like(embedding).cuda()
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "box cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                    offset_embedding += self.func(r_offset_embedding)
                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))

        return embedding, offset_embedding, idx

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict,
                    text=None, entity_embeddings=None):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_batch_queries_text_input_ids, all_batch_queries_text_attention_mask = [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure],
                                                                    query_structure),
                                         self.transform_union_structure(query_structure),
                                         0)

                if text != None:
                    input_text = {
                        'input_ids': text['batch_queries_text_input_ids_dict'][query_structure].squeeze(0),
                        'attention_mask': text['batch_queries_text_attention_mask_dict'][query_structure].squeeze(0)}
                    know_inject_query_embeddings = self.know_injector(
                        input_pattern='query',
                        input_embeddings=torch.concat([center_embedding, offset_embedding], dim=-1), text=input_text
                    )
                    center_embedding, offset_embedding = torch.chunk(know_inject_query_embeddings, 2, dim=-1)
                    offset_embedding = self.func(offset_embedding)

                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure],
                                                                             query_structure,
                                                                             0)

                if text != None:
                    all_batch_queries_text_input_ids.append(text['batch_queries_text_input_ids_dict'][query_structure])
                    all_batch_queries_text_attention_mask.append(
                        text['batch_queries_text_attention_mask_dict'][query_structure]
                    )

                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)

            if text != None:
                all_batch_queries_text_input_ids = torch.cat(all_batch_queries_text_input_ids, dim=0)
                all_batch_queries_text_attention_mask = torch.cat(all_batch_queries_text_attention_mask, dim=0)
                input_text = {'input_ids': all_batch_queries_text_input_ids,
                              'attention_mask': all_batch_queries_text_attention_mask}
                know_inject_query_embeddings = self.know_injector(
                    input_pattern='query',
                    input_embeddings=torch.concat([all_center_embeddings, all_offset_embeddings], dim=-1).squeeze(1),
                    text=input_text
                ).unsqueeze(1)
                all_center_embeddings, all_offset_embeddings = torch.chunk(know_inject_query_embeddings, 2, dim=-1)
                all_offset_embeddings = self.func(all_offset_embeddings)

        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0] // 2, 2,
                                                                           1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_box(positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings,
                                                          all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings,
                                                          all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def embed_query_vec(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_vec(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict,
                    text=None, entity_embeddings=None):
        all_center_embeddings, all_idxs = [], []
        all_batch_queries_text_input_ids, all_batch_queries_text_attention_mask = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(
                    self.transform_union_query(batch_queries_dict[query_structure],
                                               query_structure),
                    self.transform_union_structure(query_structure), 0)

                if text != None:
                    input_text = {
                        'input_ids': text['batch_queries_text_input_ids_dict'][query_structure].squeeze(0),
                        'attention_mask': text['batch_queries_text_attention_mask_dict'][query_structure].squeeze(0)}
                    know_inject_query_embeddings = self.know_injector(
                        input_pattern='query',
                        input_embeddings=center_embedding, text=input_text
                    )
                    center_embedding = know_inject_query_embeddings

                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(batch_queries_dict[query_structure], query_structure, 0)

                if text != None:
                    all_batch_queries_text_input_ids.append(text['batch_queries_text_input_ids_dict'][query_structure])
                    all_batch_queries_text_attention_mask.append(
                        text['batch_queries_text_attention_mask_dict'][query_structure]
                    )

                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)

            if text != None:
                all_batch_queries_text_input_ids = torch.cat(all_batch_queries_text_input_ids, dim=0)
                all_batch_queries_text_attention_mask = torch.cat(all_batch_queries_text_attention_mask, dim=0)
                input_text = {'input_ids': all_batch_queries_text_input_ids,
                              'attention_mask': all_batch_queries_text_attention_mask}
                know_inject_query_embeddings = self.know_injector(
                    input_pattern='query',
                    input_embeddings=all_center_embeddings.squeeze(1),
                    text=input_text
                ).unsqueeze(1)
                all_center_embeddings = know_inject_query_embeddings

        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_vec(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_vec(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_vec(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs#, all_center_embeddings

    def embed_query_beta(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using BetaE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[0], idx)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1. / embedding
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(queries, query_structure[i], idx)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list),
                                                              torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict,
                     text=None, entity_embeddings=None):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_batch_queries_text_input_ids, all_batch_queries_text_attention_mask = [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_beta(self.transform_union_query(batch_queries_dict[query_structure],
                                                                     query_structure),
                                          self.transform_union_structure(query_structure),
                                          0)

                if text != None:
                    input_text = {
                        'input_ids': text['batch_queries_text_input_ids_dict'][query_structure].squeeze(0),
                        'attention_mask': text['batch_queries_text_attention_mask_dict'][query_structure].squeeze(0)}
                    know_inject_query_embeddings = self.know_injector(
                        input_pattern='query',
                        input_embeddings=torch.concat([alpha_embedding, beta_embedding], dim=-1), text=input_text
                    )
                    # know_inject_query_embeddings = self.entity_regularizer(know_inject_query_embeddings)
                    alpha_embedding, beta_embedding = torch.chunk(know_inject_query_embeddings, 2, dim=-1)

                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure],
                                                                           query_structure,
                                                                           0)
                if text != None:
                    all_batch_queries_text_input_ids.append(text['batch_queries_text_input_ids_dict'][query_structure])
                    all_batch_queries_text_attention_mask.append(
                        text['batch_queries_text_attention_mask_dict'][query_structure]
                    )

                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)

            if text != None:
                all_batch_queries_text_input_ids = torch.cat(all_batch_queries_text_input_ids, dim=0)
                all_batch_queries_text_attention_mask = torch.cat(all_batch_queries_text_attention_mask, dim=0)
                input_text = {'input_ids': all_batch_queries_text_input_ids,
                              'attention_mask': all_batch_queries_text_attention_mask}
                know_inject_query_embeddings = self.know_injector(
                    input_pattern='query',
                    input_embeddings=torch.concat([all_alpha_embeddings, all_beta_embeddings], dim=-1).squeeze(1),
                    text=input_text
                ).unsqueeze(1)
                # know_inject_query_embeddings = self.entity_regularizer(know_inject_query_embeddings)
                all_alpha_embeddings, all_beta_embeddings = torch.chunk(know_inject_query_embeddings, 2, dim=-1)

            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)

        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0] // 2, 2, 1,
                                                                         -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0] // 2, 2, 1,
                                                                       -1)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[
                    all_idxs]  # positive samples for non-union queries in this batch
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular)).unsqueeze(1)

                # if text != None:
                #     input_text = {'input_ids': text['positive_sample_text_input_ids_dict'][all_idxs],
                #                   'attention_mask': text['positive_sample_text_attention_mask_dict'][all_idxs]}
                #     positive_embedding = self.know_injector(input_pattern='positive',
                #                                             input_embeddings=positive_embedding.squeeze(1),
                #                                             text=input_text).unsqueeze(1)

                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[
                    all_union_idxs]  # positive samples for union queries in this batch
                positive_embedding = self.entity_regularizer(
                    torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(
                        1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                if entity_embeddings is None:
                    negative_embedding = self.entity_regularizer(
                        torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(
                            batch_size, negative_size, -1))

                    # if text != None:
                    #     input_text = {
                    #         'input_ids': text['negative_sample_text_input_ids_dict'][all_idxs].view(
                    #             batch_size * negative_size, -1),
                    #         'attention_mask': text['negative_sample_text_attention_mask_dict'][all_idxs].view(
                    #             batch_size * negative_size, -1)}
                    #     negative_embedding = self.know_injector(input_pattern='negative',
                    #                                             input_embeddings=negative_embedding,
                    #                                             text=input_text).view(batch_size, negative_size, -1)
                else:
                    negative_embedding = entity_embeddings[negative_sample_regular]
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                if entity_embeddings is None:
                    negative_embedding = self.entity_regularizer(
                        torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(
                            batch_size, 1, negative_size, -1))
                else:
                    negative_embedding = entity_embeddings[negative_sample_union].unsqueeze(0)
                negative_union_logit = self.cal_logit_beta(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs#, torch.concat([all_alpha_embeddings, all_beta_embeddings], dim=-1)

    def embed_query_cone(self, queries, query_structure, idx):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                axis_entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)

                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
            else:
                axis_embedding, arg_embedding, idx = self.embed_query_cone(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):
                # negation
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)

                # projection
                else:
                    axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                    arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])

                    axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                    arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)

                    axis_r_embedding = convert_to_axis(axis_r_embedding)
                    arg_r_embedding = convert_to_axis(arg_r_embedding)

                    axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding, axis_r_embedding, arg_r_embedding)
                idx += 1
        else:
            # intersection
            axis_embedding_list = []
            arg_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, idx = self.embed_query_cone(queries, query_structure[i], idx)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)

            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)

        return axis_embedding, arg_embedding, idx

    # implement distance function
    def cal_logit_cone(self, entity_embedding, query_axis_embedding, query_arg_embedding):
        delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
        delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)

        distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
        distance_base = torch.abs(torch.sin(query_arg_embedding / 2))

        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance_in = torch.min(distance2axis, distance_base)

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cone_center_reg * torch.norm(distance_in, p=1, dim=-1)
        logit = self.gamma - distance * self.modulus

        return logit

    # implement formatting forward method
    def forward_cone(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict,
                     text=None, entity_embeddings=None):
        all_idxs, all_axis_embeddings, all_arg_embeddings = [], [], []
        all_batch_queries_text_input_ids, all_batch_queries_text_attention_mask = [], []
        all_union_idxs, all_union_axis_embeddings, all_union_arg_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                axis_embedding, arg_embedding, _ = \
                    self.embed_query_cone(self.transform_union_query(batch_queries_dict[query_structure],
                                                                     query_structure),
                                          self.transform_union_structure(query_structure), 0)

                if text != None:
                    input_text = {
                        'input_ids': text['batch_queries_text_input_ids_dict'][query_structure].squeeze(0),
                        'attention_mask': text['batch_queries_text_attention_mask_dict'][query_structure].squeeze(0)}
                    know_inject_query_embeddings = self.know_injector(
                        input_pattern='query',
                        input_embeddings=torch.concat([axis_embedding, arg_embedding], dim=-1), text=input_text
                    )
                    axis_embedding, arg_embedding = torch.chunk(know_inject_query_embeddings, 2, dim=-1)
                    axis_embedding = torch.clamp(axis_embedding, -pi, pi)
                    arg_embedding = torch.clamp(arg_embedding, 0, 2 * pi)

                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_axis_embeddings.append(axis_embedding)
                all_union_arg_embeddings.append(arg_embedding)
            else:
                axis_embedding, arg_embedding, _ = self.embed_query_cone(batch_queries_dict[query_structure], query_structure, 0)

                if text != None:
                    all_batch_queries_text_input_ids.append(text['batch_queries_text_input_ids_dict'][query_structure])
                    all_batch_queries_text_attention_mask.append(
                        text['batch_queries_text_attention_mask_dict'][query_structure]
                    )

                all_idxs.extend(batch_idxs_dict[query_structure])
                all_axis_embeddings.append(axis_embedding)
                all_arg_embeddings.append(arg_embedding)

        if len(all_axis_embeddings) > 0:
            all_axis_embeddings = torch.cat(all_axis_embeddings, dim=0).unsqueeze(1)
            all_arg_embeddings = torch.cat(all_arg_embeddings, dim=0).unsqueeze(1)

            if text != None:
                all_batch_queries_text_input_ids = torch.cat(all_batch_queries_text_input_ids, dim=0)
                all_batch_queries_text_attention_mask = torch.cat(all_batch_queries_text_attention_mask, dim=0)
                input_text = {'input_ids': all_batch_queries_text_input_ids,
                              'attention_mask': all_batch_queries_text_attention_mask}
                know_inject_query_embeddings = self.know_injector(
                    input_pattern='query',
                    input_embeddings=torch.concat([all_axis_embeddings, all_arg_embeddings], dim=-1).squeeze(1),
                    text=input_text
                ).unsqueeze(1)
                all_axis_embeddings, all_arg_embeddings = torch.chunk(know_inject_query_embeddings, 2, dim=-1)
                all_axis_embeddings = torch.clamp(all_axis_embeddings, -pi, pi)
                all_arg_embeddings = torch.clamp(all_arg_embeddings, 0, 2 * pi)

        if len(all_union_axis_embeddings) > 0:
            all_union_axis_embeddings = torch.cat(all_union_axis_embeddings, dim=0).unsqueeze(1)
            all_union_arg_embeddings = torch.cat(all_union_arg_embeddings, dim=0).unsqueeze(1)
            all_union_axis_embeddings = all_union_axis_embeddings.view(
                all_union_axis_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_arg_embeddings = all_union_arg_embeddings.view(
                all_union_arg_embeddings.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_logit = self.cal_logit_cone(positive_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)


            if len(all_union_axis_embeddings) > 0:
                # positive samples for union queries in this batch
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1)

                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_union_logit = self.cal_logit_cone(positive_embedding, all_union_axis_embeddings, all_union_arg_embeddings)

                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_logit = self.cal_logit_cone(negative_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_axis_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_union_logit = self.cal_logit_cone(negative_embedding, all_union_axis_embeddings, all_union_arg_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        batch = next(train_iterator)
        positive_sample = batch['positive_sample']
        negative_sample = batch['negative_sample']
        subsampling_weight = batch['subsample_weight']
        batch_queries = batch['query']
        query_structures = batch['query_structure']

        query_text_input = batch['query_text']
        # positive_sample_text = batch['positive_sample_text']
        # negative_sample_text = batch['negative_sample_text']

        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)

        batch_queries_text_input_ids_dict = collections.defaultdict(list)
        batch_queries_text_attention_mask_dict = collections.defaultdict(list)
        # positive_sample_text_input_ids_dict = positive_sample_text['input_ids']
        # positive_sample_text_attention_mask_dict = positive_sample_text['attention_mask']
        # negative_sample_text_input_ids_dict = negative_sample_text['input_ids']
        # negative_sample_text_attention_mask_dict = negative_sample_text['attention_mask']

        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)

            batch_queries_text_input_ids_dict[query_structures[i]].append(query_text_input['input_ids'][i].unsqueeze(0))
            batch_queries_text_attention_mask_dict[query_structures[i]].append(
                query_text_input['attention_mask'][i].unsqueeze(0))

        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                batch_queries_text_input_ids_dict[query_structure] = torch.LongTensor(
                    torch.concat(batch_queries_text_input_ids_dict[query_structure], dim=0)
                ).cuda()
                batch_queries_text_attention_mask_dict[query_structure] = torch.LongTensor(
                    torch.concat(batch_queries_text_attention_mask_dict[query_structure], dim=0)
                ).cuda()

            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                batch_queries_text_input_ids_dict[query_structure] = torch.LongTensor(
                    batch_queries_text_input_ids_dict[query_structure]
                )
                batch_queries_text_attention_mask_dict[query_structure] = torch.LongTensor(
                    batch_queries_text_attention_mask_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            # positive_sample_text_input_ids_dict = positive_sample_text_input_ids_dict.cuda()
            # positive_sample_text_attention_mask_dict = positive_sample_text_attention_mask_dict.cuda()
            # negative_sample_text_input_ids_dict = negative_sample_text_input_ids_dict.cuda()
            # negative_sample_text_attention_mask_dict = negative_sample_text_attention_mask_dict.cuda()

        text = {
            'batch_queries_text_input_ids_dict': batch_queries_text_input_ids_dict,
            'batch_queries_text_attention_mask_dict': batch_queries_text_attention_mask_dict,
            # 'positive_sample_text_input_ids_dict': positive_sample_text_input_ids_dict,
            # 'positive_sample_text_attention_mask_dict': positive_sample_text_attention_mask_dict,
            # 'negative_sample_text_input_ids_dict': negative_sample_text_input_ids_dict,
            # 'negative_sample_text_attention_mask_dict': negative_sample_text_attention_mask_dict
        }

        # text = None

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample,
                                                                      subsampling_weight, batch_queries_dict,
                                                                      batch_idxs_dict, text)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            # entity2text_inputs = test_dataloader.dataset.entity2text_inputs
            # entity_embeddings = []
            # logging.info('getting entity embeddings from know_injector')
            # for i in tqdm(range(len(entity2text_inputs))):
            #     # entity_text_inputs = test_dataloader.dataset.tokenizer(entity2text[i], max_length=args.max_seq_len)
            #     text = {'input_ids': torch.tensor(entity2text_inputs[i]['input_ids']).unsqueeze(0).cuda(),
            #             'attention_mask': torch.tensor(entity2text_inputs[i]['attention_mask']).unsqueeze(0).cuda()}
            #
            #     entity_embedding = model.know_injector(input_pattern='entity',
            #                                            input_embeddings=model.entity_regularizer(model.entity_embedding[i].unsqueeze(0)),
            #                                            text=text)
            #     entity_embeddings.append(entity_embedding)
            # entity_embeddings = torch.concat(entity_embeddings, dim=0)
            # # print()

            # query_embedding_dict = {}
            # query_name_dict = {('e', ('r',)): 0,
            #                    ('e', ('r', 'r')): 1,
            #                    ('e', ('r', 'r', 'r')): 2,
            #                    (('e', ('r',)), ('e', ('r',))): 3,
            #                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): 4}

            for batch in tqdm(test_dataloader, disable=not args.print_on_screen):

                negative_sample = batch['negative_sample']
                queries = batch['query']
                queries_unflatten = batch['query_unflatten']
                query_structures = batch['query_structure']

                # if query_name_dict[query_structures[0]] not in query_embedding_dict:
                #     query_embedding_dict[query_name_dict[query_structures[0]]] = []

                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)

                query_text_input = batch['query_text']
                batch_queries_text_input_ids_dict = collections.defaultdict(list)
                batch_queries_text_attention_mask_dict = collections.defaultdict(list)

                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)

                    batch_queries_text_input_ids_dict[query_structures[i]].append(
                        query_text_input['input_ids'][i].unsqueeze(0))
                    batch_queries_text_attention_mask_dict[query_structures[i]].append(
                        query_text_input['attention_mask'][i].unsqueeze(0))

                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                        batch_queries_text_input_ids_dict[query_structure] = torch.LongTensor(
                            torch.concat(batch_queries_text_input_ids_dict[query_structure], dim=0)
                        ).cuda()
                        batch_queries_text_attention_mask_dict[query_structure] = torch.LongTensor(
                            torch.concat(batch_queries_text_attention_mask_dict[query_structure], dim=0)
                        ).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                text = {
                    'batch_queries_text_input_ids_dict': batch_queries_text_input_ids_dict,
                    'batch_queries_text_attention_mask_dict': batch_queries_text_attention_mask_dict
                }

                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict, text, entity_embeddings=None)



                # if len(hard_answers[queries_unflatten[0]]) == 1:
                #     gt = float(list(hard_answers[queries_unflatten[0]])[0])
                #     query_embedding_dict[query_name_dict[query_structures[0]]].append(
                #         [all_center_embeddings.squeeze(0).cpu().tolist()[0] + [gt]])

                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort,
                                               model.batch_entity_range)  # achieve the ranking of all entities
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities

                for idx, (i, query, query_structure) in enumerate(
                        zip(argsort[:, 0], queries_unflatten, query_structures)):

                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()


                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        # query_embedding_with_type = []
        # for key in query_embedding_dict.keys():
        #     query_embedding_with_type.append(np.concatenate([np.concatenate(query_embedding_dict[key], axis=0),
        #                                                      np.reshape([len(query_embedding_dict[key]) * [key]],
        #                                                                 (-1, 1))], axis=-1))
        # query_embedding_with_type = np.concatenate(query_embedding_with_type, axis=0)
        # np.save('query_embedding_with_type_BetaE_QIPP', query_embedding_with_type)

        return metrics

class KGFuzzyReasoning(nn.Module):
    def __init__(
            self, nentity, nrelation, hidden_dim, gamma,
            geo, test_batch_size=1,
            box_mode=None, use_cuda=False,
            query_name_dict=None, beta_mode=None,
            logic_type='product',
            regularizer_setting=None,
            gamma_coff=20,
            loss_type='cos',
            margin_type='logsigmoid',
            device=None,
            godel_gumbel_beta=0.01,
            gumbel_temperature=1,
            projection_type='mlp',
            args=None

    ):
        super(KGFuzzyReasoning, self).__init__()

        self.device = device

        # embedding
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).to(self.device)

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.no_anchor_reg = args.fuzzy_no_anchor_reg

        if args.fuzzy_load_pretrained == True:
            with open('./trained_models/NELL-entity-emb.pt', 'rb') as f:
                # use pretrained embeddings to initialize and speed up training
                entity_embs = pickle.load(f)
            self.entity_embedding = nn.Parameter(entity_embs)
        else:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            if self.no_anchor_reg:
                nn.init.xavier_uniform_(self.entity_embedding)
            else:
                # embedding definition
                # embedding initialization
                nn.init.uniform_(tensor=self.entity_embedding, a=0, b=1)

        self.simplE = args.fuzzy_simplE
        if args.fuzzy_simplE:  # use separate head and tail embeddings for entities
            self.entity_head_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(tensor=self.entity_embedding, a=0, b=1)

        # loss
        self.gamma_coff = gamma_coff
        self.loss_type = loss_type
        self.margin_type = margin_type
        if self.loss_type == 'weighted_dot':
            self.dim_weight = nn.Parameter(torch.ones((self.entity_dim,)))
            self.dim_weight_softmax = nn.Softmax(dim=-1)

        if margin_type == 'softmax':
            self.softmax_weight = torch.Tensor([10]).to(device)

        # regularizer: how to turn elements into 0,1
        self.entity_regularizer = get_regularizer(regularizer_setting, self.entity_dim, neg_input_possible=True,
                                                  entity=True)

        # wandb.log({'loss_type': loss_type})

        self.godel_gumbel_beta = godel_gumbel_beta

        # intersection and projectizAaz<>on
        projection_dim, num_layers = beta_mode
        self.projection_net = FuzzProjection(
            nrelation,
            self.entity_dim,
            logic_type,
            regularizer_setting,
            self.relation_dim,
            projection_dim,
            num_layers,
            projection_type,
            num_rel_base=args.fuzzy_num_rel_base
        )

        self.conjunction_net = FuzzConjunction(self.entity_dim, logic_type, regularizer_setting,
                                               use_attention=args.fuzzy_use_attention, godel_gumbel_beta=godel_gumbel_beta)
        self.disjunction_net = FuzzDisjunction(self.entity_dim, logic_type, regularizer_setting,
                                               godel_gumbel_beta=godel_gumbel_beta)
        self.negation_net = FuzzNegation(self.entity_dim, logic_type, regularizer_setting)

        # gumbel softmax
        self.gumbel_temperature = gumbel_temperature  # used if loss_type == 'gumbel_softmax'
        self.gumbel_attention = args.fuzzy_gumbel_attention if args.fuzzy_gumbel_attention != 'none' else None  # None or 'plain' or 'query_dependent'
        if self.loss_type == 'gumbel_softmax' and args.fuzzy_gumbel_attention:
            self.n_distribution = self.entity_regularizer.get_num_distributions()
            self.distribution_weights = nn.Parameter(torch.ones(self.n_distribution))
            if args.fuzzy_gumbel_attention == 'query_dependent':
                self.attention_layer = nn.Linear(self.entity_dim, self.n_distribution)
        self.gumbel_query_unnorm = args.fuzzy_query_unnorm

        self.in_batch_negative = args.fuzzy_in_batch_negative

        if self.loss_type == 'dot_layernorm_digits':
            self.entity_ln = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
            self.query_ln = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

        self.counter_for_neg = args.fuzzy_with_counter  # add \neg q to negative samples

        self.margin_type = args.fuzzy_margin_type

        self.know_injector = KnowInjector(args=args)

    def forward(
            self,
            positive_sample,
            negative_sample,
            subsampling_weight,
            batch_queries_dict,
            batch_idxs_dict,
            idxs,
            inference=False,  # for discrete, use soft for training and hard for inference
            text=None
    ):

        all_idxs, all_embeddings = [], []
        all_batch_queries_text_input_ids, all_batch_queries_text_attention_mask = [], []
        all_union_idxs, all_union_embeddings = [], []
        for query_structure in batch_queries_dict:
            if 'u' in query_name_dict[query_structure] and 'DNF' in query_name_dict[query_structure]:
                query_embedding, _ = self.embed_query_fuzzy(
                    self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                    self.transform_union_structure(query_structure), 0
                )

                if text != None:
                    input_text = {
                        'input_ids': text['batch_queries_text_input_ids_dict'][query_structure].squeeze(0),
                        'attention_mask': text['batch_queries_text_attention_mask_dict'][query_structure].squeeze(0)}
                    query_embedding = self.know_injector(
                        input_pattern='query',
                        input_embeddings=query_embedding, text=input_text
                    )

                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_embeddings.append(query_embedding)
            else:
                query_embedding, _ = self.embed_query_fuzzy(batch_queries_dict[query_structure], query_structure, 0)

                if text != None:
                    all_batch_queries_text_input_ids.append(text['batch_queries_text_input_ids_dict'][query_structure])
                    all_batch_queries_text_attention_mask.append(
                        text['batch_queries_text_attention_mask_dict'][query_structure]
                    )

                all_idxs.extend(batch_idxs_dict[query_structure])
                all_embeddings.append(query_embedding)

        if len(all_union_embeddings) > 0:
            all_union_embeddings = torch.cat(all_union_embeddings, dim=0).unsqueeze(1)
            all_embeddings = self.entity_regularizer(all_union_embeddings)
            all_embeddings = self.disjunction_net(all_embeddings)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0).unsqueeze(1)
            if text != None:
                all_batch_queries_text_input_ids = torch.cat(all_batch_queries_text_input_ids, dim=0)
                all_batch_queries_text_attention_mask = torch.cat(all_batch_queries_text_attention_mask, dim=0)
                input_text = {'input_ids': all_batch_queries_text_input_ids,
                              'attention_mask': all_batch_queries_text_attention_mask}
                all_embeddings = self.know_injector(
                    input_pattern='query',
                    input_embeddings=all_embeddings.squeeze(1),
                    text=input_text
                ).unsqueeze(1)
                all_embeddings = self.entity_regularizer(all_embeddings)

        all_idxs = np.concatenate([batch_idxs_dict[query_structure] for query_structure in batch_queries_dict],
                                  axis=None)
        all_idxs = torch.LongTensor(all_idxs).to(negative_sample.device)

        #
        # if len(all_embeddings) > 0:
        #     all_embeddings = torch.cat(all_embeddings, dim=0).unsqueeze(1)

        if subsampling_weight is not None:
            subsampling_weight = subsampling_weight[all_idxs]

        if positive_sample is not None:
            if len(all_embeddings) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]
                if self.loss_type.startswith('discrete'):
                    # soft discretization
                    # use steep sigmoid to make entries closer to 0,1
                    positive_embedding = self.entity_regularizer.soft_discretize(
                        torch.index_select(
                            self.entity_embedding,
                            dim=0,
                            index=positive_sample_regular
                        ).unsqueeze(1)
                    )
                else:
                    positive_embedding = self.entity_regularizer(
                        torch.index_select(
                            self.entity_embedding,
                            dim=0,
                            index=positive_sample_regular
                        ).unsqueeze(1)
                    )

                positive_score = self.cal_logit_fuzzy(positive_embedding, all_embeddings, inference=inference)
            else:
                positive_score = torch.Tensor([]).to(self.device)

        else:
            positive_score = None

        if negative_sample is None:
            negative_score = None
        else:
            if len(all_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]

                batch_size, negative_size = negative_sample_regular.shape
                if self.loss_type.startswith('discrete'):
                    # soft discretization
                    # use steep sigmoid to make entries closer to 0,1
                    negative_embedding = self.entity_regularizer.soft_discretize(
                        torch.index_select(
                            self.entity_embedding,
                            dim=0,
                            index=negative_sample_regular.view(-1)
                        ).view(
                            batch_size,
                            negative_size,
                            -1
                        )
                    )
                else:
                    negative_embedding = self.entity_regularizer(
                        torch.index_select(
                            self.entity_embedding,
                            dim=0,
                            index=negative_sample_regular.view(-1)
                        ).view(
                            batch_size,
                            negative_size,
                            -1
                        )
                    )
                # random negative samples
                negative_score = self.cal_logit_fuzzy(negative_embedding, all_embeddings, inference=inference)
            else:
                negative_score = torch.Tensor([]).to(self.entity_embedding.device)

        if self.counter_for_neg and (not inference):
            # add \neg q as a negative sample into training
            emphasize = 16
            neg_q_embeddings = self.negation_net(all_embeddings)
            negative_score_2 = self.cal_logit_fuzzy(positive_embedding, neg_q_embeddings,
                                                    inference=inference)  # [batch_size, 1]
            negative_score_2 = negative_score_2.expand(-1, emphasize)
            negative_score = torch.cat((negative_score, negative_score_2), dim=1)
            return positive_score, negative_score, subsampling_weight, all_idxs
        else:
            return positive_score, negative_score, subsampling_weight, all_idxs

    def embed_query_fuzzy(self, queries, query_structure, idx):
        """
        :param query_structure: e.g. ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))
        :param queries: Tensor. shape [batch_size, M],
            where M is the number of elements in query_structure (6 in the above examples)
        :param idx: which column to start in tensor queries
        """
        all_relation_flag = True
        for ele in query_structure[-1]:
            # whether the current query tree has merged to one branch
            # and only need to do relation traversal,
            # e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:  # only relation traversal
            if query_structure[0] == 'e':
                if self.simplE:
                    # use head embeddings
                    embedding = self.entity_regularizer(
                        torch.index_select(self.entity_head_embedding, dim=0, index=queries[:, idx])
                    )
                else:
                    if self.no_anchor_reg:
                        # entity embedding
                        embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])

                    else:
                        # entity embedding
                        embedding = self.entity_regularizer(
                            torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                        )

                idx += 1  # move to next element (next column in queries)
            else:
                # recursion
                embedding, idx = self.embed_query_fuzzy(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):  # query_structure[-1]: ('r', 'n', 'r', ...'r')
                if query_structure[-1][i] == 'n':  # negation
                    assert (queries[:, idx] == -2).all()
                    # embedding = self.fuzzy_logic.negation(embedding)
                    embedding = self.negation_net(embedding)
                else:
                    rel_indices = queries[:, idx]
                    # embedding = self.fuzzy_logic.projection(embedding, r_embedding)
                    embedding = self.projection_net(embedding, rel_indices)
                idx += 1
        else:
            subtree_embedding_list = []
            if 'u' in query_structure[-1]:  # last one is ('u')
                # # aggregation by disjunction (union)
                # num_subtrees = len(query_structure) - 1  # last one is 'u'
                # # agg_net = self.fuzzy_logic.disjunction
                # agg_net = self.disjunction_net
                raise ValueError('Please decompose the union query into multiple path queries. For example 2u->(1p, 1p), up->(2p, 2p)')
            else:
                # aggregation by conjunction (intersection)
                num_subtrees = len(query_structure)
                agg_net = self.conjunction_net

            for i in range(num_subtrees):
                subtree_embedding, idx = self.embed_query_fuzzy(queries, query_structure[i], idx)
                subtree_embedding_list.append(subtree_embedding)

            embedding = agg_net(torch.stack(subtree_embedding_list))

            if 'u' in query_structure[-1]:  # move to next
                idx += 1

        return embedding, idx

    def get_distribution_attention(self, query_embedding=None):
        # for gumbel softmax
        softmax = nn.Softmax(dim=-1)

        if self.gumbel_attention == 'plain':
            return softmax(self.distribution_weights)
        elif self.gumbel_attention == 'query_dependent':
            distribution_attention = softmax(self.attention_layer(query_embedding))
            return distribution_attention

    def cal_logit_fuzzy(self, entity_embedding, query_embedding, inference=False):
        """
        define scoring function for loss
        :param entity_embedding: shape [batch_size, 1, dim] (positive), [batch_size, num_neg, dim] (negative)
        :param query_embedding:　shape [batch_size, 1, dim]
        :param inference: for discrete case, use soft for training and hard for inference
        :return score: shape [batch_size, 1] for positive, [batch_size, num_neg] for negative
        """
        cos = nn.CosineSimilarity(dim=-1)
        if self.loss_type == 'gumbel_softmax':  # regularizer must start with 'matrix'
            # entity embedding has been normalized
            # query embedding has been normalized as summing up to 1 if it's out of projection
            # not necessarily summing up to 1 if out of logic operations

            if self.gumbel_query_unnorm:
                query_normalized = query_embedding
            else:
                query_normalized = self.entity_regularizer.L1_normalize(query_embedding)  # vector shape

            # query_normalized = query_embedding # vector shape
            if inference:
                # hard discrete
                entity_one_hot = self.entity_regularizer.hard_discretize(entity_embedding)  # vector shape
            else:
                # convert entity to one-hot vector using gumbel
                entity_one_hot = self.entity_regularizer.soft_discretize(entity_embedding, self.gumbel_temperature)

            if self.gumbel_attention:
                entity_one_hot = self.entity_regularizer.reshape_to_matrix(entity_one_hot)
                query_normalized = self.entity_regularizer.reshape_to_matrix(query_normalized)
                score = cos(entity_one_hot, query_normalized)
                distribution_attention = self.get_distribution_attention(query_embedding)
                score = torch.sum(score * distribution_attention, dim=-1)
            else:
                # equivalent to torch.sum(entity_one_hot, query_normalized)/constant
                #    since ||entity_one_hot|| is the same for all entities
                score = cos(entity_one_hot, query_normalized)
            return score

        if self.loss_type == 'dot':
            # score = torch.sum(entity_embedding * query_embedding, dim=-1) / math.sqrt(self.entity_dim)  # dot product
            score = torch.sum(entity_embedding * query_embedding, dim=-1)  # dot product
        elif self.loss_type == 'weighted_dot':
            dim_weights = self.dim_weight_softmax(self.dim_weight)
            score = torch.sum(entity_embedding * query_embedding * dim_weights, dim=-1)
        elif self.loss_type.startswith('discrete'):
            # entity embedding should have been discretized

            if self.loss_type == 'discrete_cos':
                cos = nn.CosineSimilarity(dim=-1)
                score = cos(entity_embedding, query_embedding)
                # inference only
                # thres = 0.7
                # entity_embedding[entity_embedding >= thres] = 1
                # entity_embedding[entity_embedding < thres] = 0

            elif self.loss_type == 'discrete_prob':
                # In discrete representation, entities are considered entry value 0 or 1
                # entity_embedding should have been discretized

                # For the qth query
                # unlike other score computation, this score is not aggregated for each sample
                score = entity_embedding * query_embedding + (1 - entity_embedding) * (1 - query_embedding)

        elif self.loss_type == 'entropy':
            query_embedding = self.entity_regularizer.L1_normalize(query_embedding)  # vector shape

            # score = torch.mean(query_embedding * torch.log(entity_embedding+eps), dim=-1)

            # JSD
            m = torch.log2((query_embedding + entity_embedding) / 2 + 1e-9)
            dist = F.kl_div(m, query_embedding.expand(m.shape), reduction="none") \
                   + F.kl_div(m, entity_embedding, reduction="none")
            num_distributions = self.entity_regularizer.get_num_distributions()  # entity_dim // k
            dist = 0.5 * torch.sum(dist, dim=-1) / num_distributions
            score = 1 - dist

        elif self.loss_type == 'fuzzy_containment':
            # for Godel only
            # use with sigmoid regularizer
            # L1
            score = entity_embedding - torch.relu(entity_embedding - query_embedding)
            score = torch.max(score, dim=-1)
            # / torch.sum(entity_embedding, dim=-1)

        elif self.loss_type == 'weighted_fuzzy_containment':
            # for Godel only, use with sigmoid regularizer
            entity_vals, entity_val_weights = torch.chunk(entity_embedding, 2, dim=-1)
            query_vals, query_val_weights = torch.chunk(query_embedding, 2, dim=-1)
            val_weights = F.softmax(entity_val_weights * query_val_weights, dim=-1)

            score = entity_vals - torch.relu(entity_vals - query_vals)  # containment score
            score = torch.sum(score * val_weights, dim=-1) / torch.sum(entity_vals * val_weights, dim=-1)

        elif self.loss_type == 'cos_digits':  # use with logsigmoid_bpr_digits
            if not inference:
                entity_embedding = F.normalize(entity_embedding, p=2, dim=-1)
                query_embedding = F.normalize(query_embedding, p=2, dim=-1)
                score_digits = (entity_embedding * query_embedding) * self.entity_dim
                # score_digits = score_digits / norm.unsqueeze(2) * self.entity_dim
                return score_digits  # no aggregation, [batch_size, 1 or num_neg, dim]

            # # use cos for inference
            cos = nn.CosineSimilarity(dim=-1)
            score = cos(entity_embedding, query_embedding)

        elif self.loss_type == 'dot_layernorm_digits':  # use with logsigmoid_bpr_digits
            entity_embedding = self.entity_ln(entity_embedding)
            query_embedding = self.query_ln(query_embedding)
            score_digits = (entity_embedding * query_embedding)
            # score_digits = score_digits / norm.unsqueeze(2) * self.entity_dim
            if not inference:
                return score_digits  # no aggregation, [batch_size, 1 or num_neg, dim]
            # inference
            return torch.mean(score_digits, dim=-1)


        elif self.loss_type == 'L1_cos_digits':  # use with logsigmoid_avg
            entity_embedding = F.normalize(entity_embedding, p=1, dim=-1)
            query_embedding = F.normalize(query_embedding, p=1, dim=-1)
            score_digits = (entity_embedding * query_embedding) * self.entity_dim
            # score_digits = score_digits / norm.unsqueeze(2) * self.entity_dim
            if not inference:
                return score_digits  # no aggregation, [batch_size, 1 or num_neg, dim]
            # inference
            return torch.mean(score_digits, dim=-1)


        elif self.loss_type == 'soft_min_digits':
            # use with godel logic
            # entity_embedding = F.normalize(entity_embedding, p=2, dim=-1)
            # query_embedding = F.normalize(query_embedding, p=2, dim=-1)
            entity_embedding, query_embedding = torch.broadcast_tensors(entity_embedding, query_embedding)
            compare = torch.stack((entity_embedding, query_embedding))
            # a smooth way to compute min
            score_digits = -self.godel_gumbel_beta * torch.logsumexp(
                -compare / self.godel_gumbel_beta, 0
            )
            if not inference:
                return score_digits  # no aggregation, [batch_size, 1 or num_neg, dim]
            # inference, aggregated
            score = torch.mean(score_digits, dim=-1)
            # score = torch.logsumexp(-score_digits, dim=-1)

        elif self.loss_type == 'entity_multinomial_dot':
            entity_embedding = F.normalize(entity_embedding, p=1, dim=-1)
            score = torch.sum(entity_embedding * query_embedding, dim=-1)

        elif self.loss_type == 'normalized_entity_dot':
            score = torch.sum(entity_embedding * query_embedding, dim=-1)

        else:  # cos by default
            cos = nn.CosineSimilarity(dim=-1)
            score = cos(entity_embedding, query_embedding)
        return score

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    @staticmethod
    def compute_loss(model, positive_score, negative_score, subsampling_weight):
        device = model.device
        if model.margin_type == 'logsigmoid':
            # the loss of BetaE and RotatE
            # return score: shape[batch_size, 1] for positive, [batch_size, num_neg] for negative
            positive_dist = 1 - positive_score
            negative_dist = 1 - negative_score
            positive_unweighted_loss = -F.logsigmoid((model.gamma - positive_dist) * model.gamma_coff).squeeze(dim=1)
            negative_unweighted_loss = -F.logsigmoid((negative_dist - model.gamma) * model.gamma_coff).mean(dim=1)
            positive_sample_loss = (subsampling_weight * positive_unweighted_loss).sum()
            negative_sample_loss = (subsampling_weight * negative_unweighted_loss).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()
            loss = (positive_sample_loss + negative_sample_loss) / 2
            log = {
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
        elif model.margin_type == 'logsigmoid_avg':
            # use with cos_digits
            positive_dist = 1 - positive_score
            negative_dist = 1 - negative_score
            positive_unweighted_loss = -torch.mean(F.logsigmoid((model.gamma - positive_dist) * model.gamma_coff),
                                                   dim=-1).squeeze(dim=1)
            negative_unweighted_loss = -torch.mean(F.logsigmoid((negative_dist - model.gamma) * model.gamma_coff),
                                                   dim=-1).mean(dim=1)
            positive_sample_loss = (subsampling_weight * positive_unweighted_loss).sum()
            negative_sample_loss = (subsampling_weight * negative_unweighted_loss).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()
            loss = (positive_sample_loss + negative_sample_loss) / 2
            log = {
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
        elif model.margin_type == 'softmax':
            # positive_score shape [batch_size, 1]
            criterion = nn.CrossEntropyLoss(reduction='none')  # keep loss for each sample
            if model.loss_type != 'discrete_prob':
                softmax_weight = 10
                scores = torch.cat([positive_score, negative_score],
                                   dim=1) * softmax_weight  # [batch_size, 1+negative_sample_size]
            else:
                # score: log(prob)
                # softmax=exp(x1)/(exp(x1)+...exp(xn))=exp(x1+exp_shift)/(exp(x1+exp_shift)+...+exp(xn+exp_shift))
                # otherwise the log scores are too small and the results are all zero
                exp_shift, _ = torch.max(positive_score, dim=-1)
                exp_shift = torch.unsqueeze(exp_shift, 1)
                positive_score = positive_score - exp_shift  # still in log scale
                negative_score = negative_score - exp_shift

                # debug only
                positive_score_real = torch.exp(positive_score)
                negative_score_real = torch.exp(negative_score)

                scores = torch.cat([positive_score, negative_score], dim=1)

            target = torch.zeros((positive_score.shape[0],), dtype=torch.long).to(device)
            loss = (criterion(scores, target) * subsampling_weight).sum()  # CrossEntropyLoss includes softmax
            loss /= subsampling_weight.sum()
            log = {'loss': loss.item()}
        elif model.margin_type == 'bpr':
            # gamma as margin
            diff = torch.relu(model.gamma + negative_score - positive_score)  # relu or softplus
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }
        elif model.margin_type == 'bpr_digits':
            # positive_score: shape [batch_size, 1, dim]  (not aggregated yet)
            # negative_score: shape [batch_size, neg_per_pos, dim]
            # gamma as margin
            diff = torch.mean(torch.relu(model.gamma + negative_score - positive_score), dim=-1)  # relu or softplus
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }
        elif model.margin_type == 'logsigmoid_bpr_digits':
            # positive_score: shape [batch_size, 1, dim]  (not aggregated yet)
            # negative_score: shape [batch_size, neg_per_pos, dim]
            diff = -F.logsigmoid(model.gamma_coff * (torch.mean(positive_score - negative_score, dim=-1)))
            # diff = torch.mean(-F.logsigmoid(model.gamma_coff*(positive_score - negative_score)), dim=-1)
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }
        elif model.margin_type == 'logsigmoid_bpr':
            # gamma as margin
            diff = -F.logsigmoid(model.gamma_coff * (positive_score - negative_score))
            # diff = torch.mean(-F.logsigmoid(model.gamma_coff*(positive_score - negative_score)), dim=-1)
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }
        elif model.margin_type == 'nll':  # negative log likelihood. used together with discrete_prob
            if model.loss_type == 'discrete_prob':
                # positive_score: shape [batch_size, 1, dim]  (not aggregated yet)
                # negative_score: shape [batch_size, neg_per_pos, dim]

                eps = 1e-4  # avoid torch.log(zero)
                log_positive_score = torch.log(positive_score + eps)
                log_negative_score = torch.log(1 - negative_score + eps)  # flip for negative samples

                # negative log likelihood
                # use torch.mean instead of torch.sum to divide by a constant (dim)
                positive_sample_loss = - torch.mean(log_positive_score, dim=-1).squeeze(dim=1)
                negative_sample_loss = - torch.mean(log_negative_score, dim=-1).mean(dim=1)
                # positive_sample_loss = -positive_score.squeeze(dim=1)
                # negative_sample_loss = -torch.log(1-torch.exp(negative_score)+eps).mean(dim=1)

                positive_sample_loss = (subsampling_weight * positive_sample_loss).sum()
                negative_sample_loss = (subsampling_weight * negative_sample_loss).sum()
                positive_sample_loss /= subsampling_weight.sum()
                negative_sample_loss /= subsampling_weight.sum()
                loss = (positive_sample_loss + negative_sample_loss)
                log = {
                    'positive_sample_loss': positive_sample_loss.item(),
                    'negative_sample_loss': negative_sample_loss.item(),
                    'loss': loss.item(),
                }

            elif model.loss_type == 'entropy':

                # version 1
                # positive_sample_loss = - positive_score.squeeze(dim=-1)
                # negative_sample_loss = negative_score.mean(dim=-1)
                # positive_sample_loss = (subsampling_weight * positive_sample_loss).sum()
                # negative_sample_loss = (subsampling_weight * negative_sample_loss).sum()
                # positive_sample_loss /= subsampling_weight.sum()
                # negative_sample_loss /= subsampling_weight.sum()
                # loss = (positive_sample_loss + negative_sample_loss)
                # log = {
                #     'positive_sample_loss': positive_sample_loss.item(),
                #     'negative_sample_loss': negative_sample_loss.item(),
                #     'loss': loss.item(),
                # }

                # # # version 2

                positive_score = positive_score.squeeze(dim=-1)
                negative_score = negative_score.mean(dim=-1)
                diff = torch.relu(model.gamma + negative_score - positive_score)
                unweighted_sample_loss = torch.mean(diff, dim=-1)
                loss = (subsampling_weight * unweighted_sample_loss).sum()
                loss /= subsampling_weight.sum()
                log = {
                    'loss': loss.item(),
                }
            else:
                raise ValueError('if margin_type is nll, loss_type must be in [discrete_prob, entropy]')
        else:
            raise ValueError('margin_type must be in [logsigmoid, logsigmoid_avg, softmax, bpr, bpr_digits, logsigmoid_bpr_digits, logsigmoid_bpr, nll]')

        return loss, log

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        """
        Adapted for multiple GPUs
        """
        # device = model.module.device
        device = model.device

        model.train()
        optimizer.zero_grad()

        batch = next(train_iterator)
        positive_sample = batch['positive_sample']
        negative_sample = batch['negative_sample']
        subsampling_weight = batch['subsample_weight']
        batch_queries = batch['query']
        query_structures = batch['query_structure']

        query_text_input = batch['query_text']
        # positive_sample_text = batch['positive_sample_text']
        # negative_sample_text = batch['negative_sample_text']

        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)

        batch_queries_text_input_ids_dict = collections.defaultdict(list)
        batch_queries_text_attention_mask_dict = collections.defaultdict(list)
        # positive_sample_text_input_ids_dict = positive_sample_text['input_ids']
        # positive_sample_text_attention_mask_dict = positive_sample_text['attention_mask']
        # negative_sample_text_input_ids_dict = negative_sample_text['input_ids']
        # negative_sample_text_attention_mask_dict = negative_sample_text['attention_mask']

        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)

            batch_queries_text_input_ids_dict[query_structures[i]].append(query_text_input['input_ids'][i].unsqueeze(0))
            batch_queries_text_attention_mask_dict[query_structures[i]].append(
                query_text_input['attention_mask'][i].unsqueeze(0))

        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                batch_queries_text_input_ids_dict[query_structure] = torch.LongTensor(
                    torch.concat(batch_queries_text_input_ids_dict[query_structure], dim=0)
                ).cuda()
                batch_queries_text_attention_mask_dict[query_structure] = torch.LongTensor(
                    torch.concat(batch_queries_text_attention_mask_dict[query_structure], dim=0)
                ).cuda()

            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                batch_queries_text_input_ids_dict[query_structure] = torch.LongTensor(
                    batch_queries_text_input_ids_dict[query_structure]
                )
                batch_queries_text_attention_mask_dict[query_structure] = torch.LongTensor(
                    batch_queries_text_attention_mask_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            # positive_sample_text_input_ids_dict = positive_sample_text_input_ids_dict.cuda()
            # positive_sample_text_attention_mask_dict = positive_sample_text_attention_mask_dict.cuda()
            # negative_sample_text_input_ids_dict = negative_sample_text_input_ids_dict.cuda()
            # negative_sample_text_attention_mask_dict = negative_sample_text_attention_mask_dict.cuda()

        text = {
            'batch_queries_text_input_ids_dict': batch_queries_text_input_ids_dict,
            'batch_queries_text_attention_mask_dict': batch_queries_text_attention_mask_dict,
            # 'positive_sample_text_input_ids_dict': positive_sample_text_input_ids_dict,
            # 'positive_sample_text_attention_mask_dict': positive_sample_text_attention_mask_dict,
            # 'negative_sample_text_input_ids_dict': negative_sample_text_input_ids_dict,
            # 'negative_sample_text_attention_mask_dict': negative_sample_text_attention_mask_dict
        }

        # text = None

        if args.cuda:
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            subsampling_weight = subsampling_weight.to(device)
            # no need to move query_structure_idxs to GPU

        # nn.DataParallel helper
        batch_size = len(positive_sample)
        slice_idxs = torch.arange(0, batch_size).view((batch_size, 1))

        # model(positive_sample, negative_sample,
        #       subsampling_weight, batch_queries_dict,
        #       batch_idxs_dict, text)

        positive_score, negative_score, subsampling_weight, _ = model(
            positive_sample,
            negative_sample,
            subsampling_weight,
            batch_queries_dict,  # np.array([queries]), won't be split when using multiple GPUs
            batch_idxs_dict,  # torch.LongTensor
            slice_idxs,  # to help track batch_queries and query_structures when using multiple GPUs
            inference=False,
            text=text
        )
        loss, log = KGFuzzyReasoning.compute_loss(model, positive_score, negative_score, subsampling_weight)

        loss.backward()
        optimizer.step()

        if model.loss_type == 'normalized_entity_dot':
            with torch.no_grad():
                # normalize entity embeddings
                normalized = nn.Parameter(torch.clamp(model.entity_embedding, 0, 1))
                # F1 normalize
                model.entity_embedding = nn.Parameter(F.normalize(normalized, p=1, dim=-1))

        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        model.eval()

        # device = model.module.device
        device = model.device

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            # for negative_sample, queries, queries_unflatten, query_structure_idxs in tqdm(test_dataloader,
            #                                                                               disable=not args.print_on_screen):
            #     # example: query_structures: [('e', ('r',))].  queries: [[1804,4]]. queries_unflatten: [(1804, (4,)]
            #     if args.cuda:
            #         negative_sample = negative_sample.to(device)

            for batch in tqdm(test_dataloader, disable=not args.print_on_screen):

                negative_sample = batch['negative_sample']
                queries = batch['query']
                queries_unflatten = batch['query_unflatten']
                query_structures = batch['query_structure']

                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)

                query_text_input = batch['query_text']
                batch_queries_text_input_ids_dict = collections.defaultdict(list)
                batch_queries_text_attention_mask_dict = collections.defaultdict(list)

                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)

                    batch_queries_text_input_ids_dict[query_structures[i]].append(
                        query_text_input['input_ids'][i].unsqueeze(0))
                    batch_queries_text_attention_mask_dict[query_structures[i]].append(
                        query_text_input['attention_mask'][i].unsqueeze(0))

                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                        batch_queries_text_input_ids_dict[query_structure] = torch.LongTensor(
                            torch.concat(batch_queries_text_input_ids_dict[query_structure], dim=0)
                        ).cuda()
                        batch_queries_text_attention_mask_dict[query_structure] = torch.LongTensor(
                            torch.concat(batch_queries_text_attention_mask_dict[query_structure], dim=0)
                        ).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                text = {
                    'batch_queries_text_input_ids_dict': batch_queries_text_input_ids_dict,
                    'batch_queries_text_attention_mask_dict': batch_queries_text_attention_mask_dict
                }

                # nn.DataParallel helper
                batch_size = len(negative_sample)
                slice_idxs = torch.arange(0, batch_size).view((batch_size, 1))

                _, negative_logit, _, idxs = model(
                    None,
                    negative_sample,
                    None,
                    batch_queries_dict,  # np.array([queries]), won't be split when using multiple GPUs
                    batch_idxs_dict,
                    slice_idxs,  # to help track batch_queries and query_structures when using multiple GPUs
                    inference=True,
                    text=text
                )

                if model.loss_type == 'discrete_prob':
                    # negative_logit shape[batch_size, num_entity, dim], not aggregated yet
                    eps = 1e-4
                    negative_logit = torch.sum(torch.log(negative_logit + eps), dim=-1)

                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort,
                                               model.batch_entity_range)  # achieve the ranking of all entities
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities

                for idx, (i, query, query_structure) in enumerate(
                        zip(argsort[:, 0], queries_unflatten, query_structures)):

                    # ans_list = list(easy_answers[query]) + list(hard_answers[query])
                    # ans = set(ans_list)
                    # hard_ans_list = list(hard_answers[query])
                    # all_idx = set(range(model.nentity))
                    # false_ans = all_idx - ans
                    # false_ans_list = list(false_ans)
                    #
                    # ans_idxs = np.array(hard_ans_list)
                    # vals = np.zeros((len(ans_idxs), model.nentity))
                    # vals[np.arange(len(ans_idxs)), ans_idxs] = 1
                    # axis2 = np.tile(false_ans_list, len(ans_idxs))
                    # axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))
                    # vals[axis1, axis2] = 1
                    # b = torch.Tensor(vals).to(negative_logit.device)
                    #
                    # filter_score = b * negative_logit[idx].unsqueeze(0)
                    #
                    # argsort = torch.argsort(filter_score, dim=1, descending=True)
                    # ans_tensor = torch.LongTensor(hard_ans_list).to(negative_logit.device)
                    #
                    # argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1)
                    # ranking = (argsort == 0).nonzero()
                    # ranking = ranking[:, 1]
                    # ranking = ranking + 1
                    #
                    # h1 = torch.mean((ranking <= 1).to(torch.float)).item()
                    # h3 = torch.mean((ranking <= 3).to(torch.float)).item()
                    # h10 = torch.mean((ranking <= 10).to(torch.float)).item()
                    # # mrm = torch.mean(ranking.to(torch.float)).item()
                    # mrr = torch.mean(1. / ranking.to(torch.float)).item()
                    # num_hard = len(hard_ans_list)
                    #
                    # logs[query_structure].append({
                    #     'MRR': mrr,
                    #     'HITS1': h1,
                    #     'HITS3': h3,
                    #     'HITS10': h10,
                    #     'num_hard_answer': num_hard,
                    # })

                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics

class MLP2Vec(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, test_batch_size=1,
                 use_cuda=False,
                 query_name_dict=None,
                 mlp_mode=None,
                 args=None):
        super(MLP2Vec, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size,
                                                                               1).cuda() if self.use_cuda else torch.arange(
            nentity).to(torch.float).repeat(test_batch_size, 1)  # used in test_step
        self.query_name_dict = query_name_dict
        self.layers = mlp_mode
        self.args = args

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))  # center for entities
        # nn.init.uniform_(
        #     tensor=self.entity_embedding,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )

        self.notNN1 = NotMLP(self.layers, self.entity_dim)
        self.projectionNN1 = ProjectionMLP(self.layers, self.entity_dim)
        self.andNN1 = AndMLP(self.layers, self.entity_dim)
        # self.orNN1 = OrMLP(self.layers, self.entity_dim)

        self.notNN2 = NotMLP(self.layers, self.entity_dim)
        self.projectionNN2 = ProjectionMLP(self.layers, self.entity_dim)
        self.andNN2 = AndMLP(self.layers, self.entity_dim)
        # self.orNN2 = OrMLP(self.layers, self.entity_dim)

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.know_injector = KnowInjector(args=self.args)

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict,
                text=None, entity_embeddings=None):

        return self.forward_mlp(positive_sample, negative_sample, subsampling_weight,
                                batch_queries_dict, batch_idxs_dict, text, entity_embeddings)

    def embed_query_mlp2vector(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[
            -1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_mlp2vector(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    # Negation
                    assert (queries[:, idx] == -2).all()
                    embedding_1 = self.notNN1(embedding)
                    embedding_2 = self.notNN2(embedding)
                    embedding = (embedding_1 + embedding_2) / 2
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding_1 = self.projectionNN1(embedding, r_embedding)
                    embedding_2 = self.projectionNN2(embedding, r_embedding)
                    embedding = (embedding_1 + embedding_2) / 2
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_mlp2vector(queries, query_structure[i], idx)
                embedding_list.append(embedding)

            vector = embedding_list[0]
            for i in range(1, len(embedding_list)):
                vector_1 = self.andNN1(vector, embedding_list[i])
                vector_2 = self.andNN2(vector, embedding_list[i])
                vector = (vector_1 + vector_2) / 2
            embedding = vector

        return embedding, idx

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]  # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1),
                                 torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0] * 2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_mlp(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_mlp(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict,
                    text=None, entity_embeddings=None):
        all_center_embeddings, all_idxs = [], []
        all_batch_queries_text_input_ids, all_batch_queries_text_attention_mask = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_mlp2vector(
                    self.transform_union_query(batch_queries_dict[query_structure],
                                               query_structure),
                    self.transform_union_structure(query_structure), 0)

                if text != None:
                    input_text = {
                        'input_ids': text['batch_queries_text_input_ids_dict'][query_structure].squeeze(0),
                        'attention_mask': text['batch_queries_text_attention_mask_dict'][query_structure].squeeze(0)}
                    know_inject_query_embeddings = self.know_injector(
                        input_pattern='query',
                        input_embeddings=center_embedding, text=input_text
                    )
                    center_embedding = know_inject_query_embeddings

                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_mlp2vector(batch_queries_dict[query_structure],
                                                                  query_structure, 0)

                if text != None:
                    all_batch_queries_text_input_ids.append(text['batch_queries_text_input_ids_dict'][query_structure])
                    all_batch_queries_text_attention_mask.append(
                        text['batch_queries_text_attention_mask_dict'][query_structure]
                    )

                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)

            if text != None:
                all_batch_queries_text_input_ids = torch.cat(all_batch_queries_text_input_ids, dim=0)
                all_batch_queries_text_attention_mask = torch.cat(all_batch_queries_text_attention_mask, dim=0)
                input_text = {'input_ids': all_batch_queries_text_input_ids,
                              'attention_mask': all_batch_queries_text_attention_mask}
                know_inject_query_embeddings = self.know_injector(
                    input_pattern='query',
                    input_embeddings=all_center_embeddings.squeeze(1),
                    text=input_text
                ).unsqueeze(1)
                all_center_embeddings = know_inject_query_embeddings

        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2,
                                                                           1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_mlp(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_mlp(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size,
                                                                                                     negative_size, -1)
                negative_logit = self.cal_logit_mlp(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = torch.index_select(self.entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1,
                                                                                                   negative_size, -1)
                negative_union_logit = self.cal_logit_mlp(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        batch = next(train_iterator)
        positive_sample = batch['positive_sample']
        negative_sample = batch['negative_sample']
        subsampling_weight = batch['subsample_weight']
        batch_queries = batch['query']
        query_structures = batch['query_structure']

        query_text_input = batch['query_text']
        # positive_sample_text = batch['positive_sample_text']
        # negative_sample_text = batch['negative_sample_text']

        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)

        batch_queries_text_input_ids_dict = collections.defaultdict(list)
        batch_queries_text_attention_mask_dict = collections.defaultdict(list)
        # positive_sample_text_input_ids_dict = positive_sample_text['input_ids']
        # positive_sample_text_attention_mask_dict = positive_sample_text['attention_mask']
        # negative_sample_text_input_ids_dict = negative_sample_text['input_ids']
        # negative_sample_text_attention_mask_dict = negative_sample_text['attention_mask']

        for i, query in enumerate(batch_queries):  # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)

            batch_queries_text_input_ids_dict[query_structures[i]].append(query_text_input['input_ids'][i].unsqueeze(0))
            batch_queries_text_attention_mask_dict[query_structures[i]].append(
                query_text_input['attention_mask'][i].unsqueeze(0))

        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                batch_queries_text_input_ids_dict[query_structure] = torch.LongTensor(
                    torch.concat(batch_queries_text_input_ids_dict[query_structure], dim=0)
                ).cuda()
                batch_queries_text_attention_mask_dict[query_structure] = torch.LongTensor(
                    torch.concat(batch_queries_text_attention_mask_dict[query_structure], dim=0)
                ).cuda()

            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                batch_queries_text_input_ids_dict[query_structure] = torch.LongTensor(
                    batch_queries_text_input_ids_dict[query_structure]
                )
                batch_queries_text_attention_mask_dict[query_structure] = torch.LongTensor(
                    batch_queries_text_attention_mask_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            # positive_sample_text_input_ids_dict = positive_sample_text_input_ids_dict.cuda()
            # positive_sample_text_attention_mask_dict = positive_sample_text_attention_mask_dict.cuda()
            # negative_sample_text_input_ids_dict = negative_sample_text_input_ids_dict.cuda()
            # negative_sample_text_attention_mask_dict = negative_sample_text_attention_mask_dict.cuda()

        text = {
            'batch_queries_text_input_ids_dict': batch_queries_text_input_ids_dict,
            'batch_queries_text_attention_mask_dict': batch_queries_text_attention_mask_dict,
            # 'positive_sample_text_input_ids_dict': positive_sample_text_input_ids_dict,
            # 'positive_sample_text_attention_mask_dict': positive_sample_text_attention_mask_dict,
            # 'negative_sample_text_input_ids_dict': negative_sample_text_input_ids_dict,
            # 'negative_sample_text_attention_mask_dict': negative_sample_text_attention_mask_dict
        }

        # text = None

        positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample,
                                                                      subsampling_weight, batch_queries_dict,
                                                                      batch_idxs_dict, text=text)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False,
                  save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(test_dataloader, disable=not args.print_on_screen):

                negative_sample = batch['negative_sample']
                queries = batch['query']
                queries_unflatten = batch['query_unflatten']
                query_structures = batch['query_structure']

                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)

                query_text_input = batch['query_text']
                batch_queries_text_input_ids_dict = collections.defaultdict(list)
                batch_queries_text_attention_mask_dict = collections.defaultdict(list)

                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)

                    batch_queries_text_input_ids_dict[query_structures[i]].append(
                        query_text_input['input_ids'][i].unsqueeze(0))
                    batch_queries_text_attention_mask_dict[query_structures[i]].append(
                        query_text_input['attention_mask'][i].unsqueeze(0))

                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(
                            batch_queries_dict[query_structure]).cuda()
                        batch_queries_text_input_ids_dict[query_structure] = torch.LongTensor(
                            torch.concat(batch_queries_text_input_ids_dict[query_structure], dim=0)
                        ).cuda()
                        batch_queries_text_attention_mask_dict[query_structure] = torch.LongTensor(
                            torch.concat(batch_queries_text_attention_mask_dict[query_structure], dim=0)
                        ).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                text = {
                    'batch_queries_text_input_ids_dict': batch_queries_text_input_ids_dict,
                    'batch_queries_text_attention_mask_dict': batch_queries_text_attention_mask_dict
                }
                if (('e', ('r', 'r', 'n')), ('e', ('r',))) in batch_queries_text_input_ids_dict:
                    print(test_dataloader.dataset.tokenizer.batch_decode(batch_queries_text_input_ids_dict[(('e', ('r', 'r', 'n')), ('e', ('r',)))]))
                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict,
                                                   text)

                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(
                        argsort) == args.test_batch_size:  # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort,
                                               model.batch_entity_range)  # achieve the ranking of all entities
                else:  # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )  # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   )  # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(
                        zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1  # filtered setting
                    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers

                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]]) / len(
                    logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics