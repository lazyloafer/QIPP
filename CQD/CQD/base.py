# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch import optim, Tensor
import math
from collections import OrderedDict
import numpy as np
import pickle
import os
from transformers import AutoTokenizer
from know_inject import KnowInjector
from CQD.util import query_to_atoms
import CQD.discrete as d2

from typing import Tuple, List, Optional, Dict


class N3:
    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class CQD(nn.Module):
    MIN_NORM = 'min'
    PROD_NORM = 'prod'
    NORMS = {MIN_NORM, PROD_NORM}

    def __init__(self,
                 nentity: int,
                 nrelation: int,
                 rank: int,
                 init_size: float = 1e-3,
                 reg_weight: float = 1e-2,
                 test_batch_size: int = 1,
                 method: str = 'discrete',
                 t_norm_name: str = 'prod',
                 k: int = 5,
                 query_name_dict: Optional[Dict] = None,
                 do_sigmoid: bool = False,
                 do_normalize: bool = False,
                 use_cuda: bool = False,
                 args = None):
        super(CQD, self).__init__()

        self.rank = rank
        self.nentity = nentity
        self.nrelation = nrelation
        self.method = method
        self.t_norm_name = t_norm_name
        self.k = k
        self.query_name_dict = query_name_dict
        self.args = args

        sizes = (nentity, nrelation)
        self.embeddings = nn.ModuleList([nn.Embedding(s, 2 * rank, sparse=False) for s in sizes[:2]])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.regularizer = N3(reg_weight)

        self.do_sigmoid = do_sigmoid
        self.do_normalize = do_normalize

        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)
        if self.use_cuda is True:
            self.batch_entity_range = self.batch_entity_range.cuda()

        self.entity2text, self.relation2text = self.get_ent_rel_text()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.plm_name)
        self.know_injector = KnowInjector(args=self.args)

        # print()

    def get_ent_rel_text(self):
        data_path = self.args.data_path
        # data_path = 'src_data/{}'.format(self.opt['dataset'])

        with open('{}/{}'.format(data_path, 'ent2id.pkl'), 'rb') as f:
            ent2id = pickle.load(f)
            print(len(ent2id))
        entity2text = {}
        with open(os.path.join(data_path, "entity2text.txt"), 'r', encoding='utf-8') as fin:
            for l in fin:
                entity, text = l.strip().split('\t')
                if entity in ent2id.keys():
                    entity2text[ent2id[entity]] = text
                else:
                    pass
        entity2text = OrderedDict(sorted(entity2text.items(), key=lambda x: x[0]))
        print(len(entity2text))

        with open('{}/{}'.format(data_path, 'rel2id.pkl'), 'rb') as f:
            rel2id = pickle.load(f)
        relation2text = {}
        with open(os.path.join(data_path, "relation2text.txt"), 'r', encoding='utf-8') as fin:
            for l in fin:
                relation, text = l.strip().split('\t')
                if 'FB15k' in data_path or 'FB15K' in data_path:
                    relation2text[rel2id['+{}'.format(relation)]] = text
                    relation2text[rel2id['-{}'.format(relation)]] = "inverse " + text
                elif 'NELL' in data_path:
                    relation2text[rel2id[relation]] = text
                # print(relation2text)
                # print()
        relation2text = OrderedDict(sorted(relation2text.items(), key=lambda x: x[0]))
        # print(relation2text)

        return np.expand_dims(np.array([entity2text[i] for i in range(len(entity2text))]), axis=-1), \
               np.expand_dims(np.array([relation2text[i] for i in range(len(relation2text))]), axis=-1)

    def collate_tokens(self, values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
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

    def split(self,
              lhs_emb: Tensor,
              rel_emb: Tensor,
              rhs_emb: Tensor,
              query_text_inputs) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        # batch_size = self.args.b

        lhs_emb_R = lhs_emb[..., :self.rank]
        lhs_emb_I = lhs_emb[..., self.rank:]

        rel_emb_R = rel_emb[..., :self.rank]
        rel_emb_I = rel_emb[..., self.rank:]

        rhs_emb_R = rhs_emb[..., :self.rank]
        rhs_emb_I = rhs_emb[..., self.rank:]

        lhs = lhs_emb_R, lhs_emb_I
        rel = rel_emb_R, rel_emb_I
        rhs = rhs_emb_R, rhs_emb_I
        return lhs, rel, rhs

    def loss(self,
             triples: Tensor) -> Tensor:
        (scores_o, scores_s), factors = self.score_candidates(triples)
        l_fit = self.loss_fn(scores_o, triples[:, 2]) + self.loss_fn(scores_s, triples[:, 0])
        l_reg = self.regularizer.forward(factors)
        return l_fit + l_reg

    def score_candidates(self,
                         triples: Tensor) -> Tuple[Tuple[Tensor, Tensor], Optional[List[Tensor]]]:
        lhs_emb = self.embeddings[0](triples[:, 0])
        rel_emb = self.embeddings[1](triples[:, 1])
        rhs_emb = self.embeddings[0](triples[:, 2])
        to_score = self.embeddings[0].weight

        text_lhs = self.entity2text[triples[:, 0].cpu()].tolist()
        text_rel = self.relation2text[triples[:, 1].cpu()].tolist()
        text_rhs = self.entity2text[triples[:, 2].cpu()].tolist()

        query_text_list = ['({}, ({}), ?)'.format(text_lhs[i][0], text_rel[i][0]) for i in range(len(text_lhs))]

        query_text_inputs1 = self.tokenizer(query_text_list)
        query_text_inputs1 = {
            'input_ids': self.collate_tokens([torch.tensor(_) for _ in query_text_inputs1['input_ids']], pad_idx=0).to(
                'cuda' if self.args.cuda else 'cpu'),
            'attention_mask': self.collate_tokens([torch.tensor(_) for _ in query_text_inputs1['attention_mask']],
                                             pad_idx=0).to('cuda' if self.args.cuda else 'cpu')
        }
        scores_o, _ = self.score_o(lhs_emb, rel_emb, to_score, query_text_inputs1)

        query_text_list = ['(?, ({}), {})'.format(text_rel[i][0], text_rhs[i][0]) for i in range(len(text_lhs))]

        query_text_inputs2 = self.tokenizer(query_text_list)
        query_text_inputs2 = {
            'input_ids': self.collate_tokens([torch.tensor(_) for _ in query_text_inputs2['input_ids']], pad_idx=0).to(
                'cuda' if self.args.cuda else 'cpu'),
            'attention_mask': self.collate_tokens([torch.tensor(_) for _ in query_text_inputs2['attention_mask']],
                                                  pad_idx=0).to('cuda' if self.args.cuda else 'cpu')
        }
        scores_s, _ = self.score_s(to_score, rel_emb, rhs_emb, query_text_inputs2)
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb, query_text_inputs=None)
        factors = self.get_factors(lhs, rel, rhs)
        return (scores_o, scores_s), factors

    def score_o(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor,
                query_text_inputs,
                return_factors: bool = False) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb, query_text_inputs)

        # lhs_emb_R = self.know_injector(
        #     input_pattern='query',
        #     input_embeddings=lhs[0], text=query_text_inputs
        # )
        # lhs_emb_I = self.know_injector(
        #     input_pattern='query',
        #     input_embeddings=lhs[1], text=query_text_inputs
        # )
        #
        # rel_emb_R = self.know_injector(
        #     input_pattern='query',
        #     input_embeddings=rel[0], text=query_text_inputs
        # )
        # rel_emb_I = self.know_injector(
        #     input_pattern='query',
        #     input_embeddings=rel[1], text=query_text_inputs
        # )
        #
        # lhs = lhs_emb_R, lhs_emb_I
        # rel = rel_emb_R, rel_emb_I

        q_embedding1 = lhs[0] * rel[0] - lhs[1] * rel[1]
        q_embedding1 = self.know_injector(
            input_pattern='query',
            input_embeddings=q_embedding1, text=query_text_inputs
        ) + q_embedding1
        q_embedding2 = lhs[1] * rel[0] + lhs[0] * rel[1]
        q_embedding2 = self.know_injector(
            input_pattern='query',
            input_embeddings=q_embedding2, text=query_text_inputs
        ) + q_embedding2
        score_1 = q_embedding1 @ rhs[0].transpose(-1, -2)
        score_2 = q_embedding2 @ rhs[1].transpose(-1, -2)
        factors = None #self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def score_s(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor,
                query_text_inputs,
                return_factors: bool = False) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb, query_text_inputs)

        # rel_emb_R = self.know_injector(
        #     input_pattern='query',
        #     input_embeddings=rel[0], text=query_text_inputs
        # )
        # rel_emb_I = self.know_injector(
        #     input_pattern='query',
        #     input_embeddings=rel[1], text=query_text_inputs
        # )
        #
        # rhs_emb_R = self.know_injector(
        #     input_pattern='query',
        #     input_embeddings=rhs[0], text=query_text_inputs
        # )
        # rhs_emb_I = self.know_injector(
        #     input_pattern='query',
        #     input_embeddings=rhs[1], text=query_text_inputs
        # )
        #
        # rel = rel_emb_R, rel_emb_I
        # rhs = rhs_emb_R, rhs_emb_I

        q_embedding1 = rhs[0] * rel[0] + rhs[1] * rel[1]
        q_embedding1 = self.know_injector(
            input_pattern='query',
            input_embeddings=q_embedding1, text=query_text_inputs
        ) + q_embedding1
        q_embedding2 = rhs[1] * rel[0] - rhs[0] * rel[1]
        q_embedding2 = self.know_injector(
            input_pattern='query',
            input_embeddings=q_embedding2, text=query_text_inputs
        ) + q_embedding2

        score_1 = q_embedding1 @ lhs[0].transpose(-1, -2)
        score_2 = q_embedding2 @ lhs[1].transpose(-1, -2)
        factors = None #self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def get_factors(self,
                    lhs: Tuple[Tensor, Tensor],
                    rel: Tuple[Tensor, Tensor],
                    rhs: Tuple[Tensor, Tensor]) -> List[Tensor]:
        factors = []
        for term in (lhs, rel, rhs):
            factors.append(torch.sqrt(term[0] ** 2 + term[1] ** 2))
        return factors

    def get_full_embeddings(self, queries: Tensor) \
            -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        lhs = rel = rhs = None
        if torch.sum(queries[:, 0]).item() > 0:
            lhs = self.embeddings[0](queries[:, 0])
        if torch.sum(queries[:, 1]).item() > 0:
            rel = self.embeddings[1](queries[:, 1])
        if torch.sum(queries[:, 2]).item() > 0:
            rhs = self.embeddings[0](queries[:, 2])
        return lhs, rel, rhs

    def batch_t_norm(self, scores: Tensor) -> Tensor:
        if self.t_norm_name == CQD.MIN_NORM:
            scores = torch.min(scores, dim=1)[0]
        elif self.t_norm_name == CQD.PROD_NORM:
            scores = torch.prod(scores, dim=1)
        else:
            raise ValueError(f't_norm must be one of {CQD.NORMS}, got {self.t_norm_name}')

        return scores

    def batch_t_conorm(self, scores: Tensor) -> Tensor:
        if self.t_norm_name == CQD.MIN_NORM:
            scores = torch.max(scores, dim=1, keepdim=True)[0]
        elif self.t_norm_name == CQD.PROD_NORM:
            scores = torch.sum(scores, dim=1, keepdim=True) - torch.prod(scores, dim=1, keepdim=True)
        else:
            raise ValueError(f't_norm must be one of {CQD.NORMS}, got {self.t_norm_name}')

        return scores

    def reduce_query_score(self, atom_scores, conjunction_mask, negation_mask):
        batch_size, num_atoms, *extra_dims = atom_scores.shape

        atom_scores = torch.sigmoid(atom_scores)
        scores = atom_scores.clone()
        scores[negation_mask] = 1 - atom_scores[negation_mask]

        disjunctions = scores[~conjunction_mask].reshape(batch_size, -1, *extra_dims)
        conjunctions = scores[conjunction_mask].reshape(batch_size, -1, *extra_dims)

        if disjunctions.shape[1] > 0:
            disjunctions = self.batch_t_conorm(disjunctions)

        conjunctions = torch.cat([disjunctions, conjunctions], dim=1)

        t_norm = self.batch_t_norm(conjunctions)
        return t_norm

    def forward(self,
                positive_sample,
                negative_sample,
                subsampling_weight,
                batch_queries_dict: Dict[Tuple, Tensor],
                batch_idxs_dict):
        all_idxs = []
        all_scores = []

        scores = None

        for query_structure, queries in batch_queries_dict.items():
            batch_size = queries.shape[0]
            atoms, num_variables, conjunction_mask, negation_mask = query_to_atoms(query_structure, queries)

            all_idxs.extend(batch_idxs_dict[query_structure])

            # [False, True]
            target_mask = torch.sum(atoms == -num_variables, dim=-1) > 0

            # Offsets identify variables across different batches
            var_id_offsets = torch.arange(batch_size, device=atoms.device) * num_variables
            var_id_offsets = var_id_offsets.reshape(-1, 1, 1)

            # Replace negative variable IDs with valid identifiers
            vars_mask = atoms < 0
            atoms_offset_vars = -atoms + var_id_offsets

            atoms[vars_mask] = atoms_offset_vars[vars_mask]

            head, rel, tail = atoms[..., 0], atoms[..., 1], atoms[..., 2]
            head_vars_mask = vars_mask[..., 0]

            with torch.no_grad():
                h_emb_constants = self.embeddings[0](head)
                r_emb = self.embeddings[1](rel)

            if 'continuous' in self.method:
                h_emb = h_emb_constants
                if num_variables > 1:
                    # var embedding for ID 0 is unused for ease of implementation
                    var_embs = nn.Embedding((num_variables * batch_size) + 1, self.rank * 2)
                    var_embs.weight.data *= self.init_size

                    var_embs.to(atoms.device)
                    optimizer = optim.Adam(var_embs.parameters(), lr=0.1)
                    prev_loss_value = 1000
                    loss_value = 999
                    i = 0

                    # CQD-CO optimization loop
                    while i < 1000 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                        prev_loss_value = loss_value

                        h_emb = h_emb_constants.clone()
                        # Fill variable positions with optimizable embeddings
                        h_emb[head_vars_mask] = var_embs(head[head_vars_mask])

                        t_emb = var_embs(tail)
                        scores, factors = self.score_o(h_emb.unsqueeze(-2),
                                                       r_emb.unsqueeze(-2),
                                                       t_emb.unsqueeze(-2),
                                                       return_factors=True)

                        query_score = self.reduce_query_score(scores,
                                                              conjunction_mask,
                                                              negation_mask)

                        loss = - query_score.mean() + self.regularizer.forward(factors)
                        loss_value = loss.item()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        i += 1

                with torch.no_grad():
                    # Select predicates involving target variable only
                    conjunction_mask = conjunction_mask[target_mask].reshape(batch_size, -1)
                    negation_mask = negation_mask[target_mask].reshape(batch_size, -1)

                    target_mask = target_mask.unsqueeze(-1).expand_as(h_emb)
                    emb_size = h_emb.shape[-1]
                    h_emb = h_emb[target_mask].reshape(batch_size, -1, emb_size)
                    r_emb = r_emb[target_mask].reshape(batch_size, -1, emb_size)
                    to_score = self.embeddings[0].weight

                    scores, factors = self.score_o(h_emb, r_emb, to_score)
                    query_score = self.reduce_query_score(scores,
                                                          conjunction_mask,
                                                          negation_mask)
                    all_scores.append(query_score)

                scores = torch.cat(all_scores, dim=0)

            elif 'discrete' in self.method:
                graph_type = self.query_name_dict[query_structure]

                def t_norm(a: Tensor, b: Tensor) -> Tensor:
                    return torch.minimum(a, b)

                def t_conorm(a: Tensor, b: Tensor) -> Tensor:
                    return torch.maximum(a, b)

                if self.t_norm_name == CQD.PROD_NORM:
                    def t_norm(a: Tensor, b: Tensor) -> Tensor:
                        return a * b

                    def t_conorm(a: Tensor, b: Tensor) -> Tensor:
                        return 1 - ((1 - a) * (1 - b))

                def normalize(scores_: Tensor) -> Tensor:
                    scores_ = scores_ - scores_.min(1, keepdim=True)[0]
                    scores_ = scores_ / scores_.max(1, keepdim=True)[0]
                    return scores_

                def scoring_function(lhs_: Tensor, rel_: Tensor, rhs_: Tensor, queries) -> Tensor:
                    text_lhs = self.entity2text[queries[:, 0].cpu()].tolist()
                    text_rel = self.relation2text[queries[:, 1].cpu()].tolist()
                    query_text_list = ['({}, ({}), ?)'.format(text_lhs[i][0], text_rel[i][0]) for i in
                                       range(len(text_lhs))]

                    query_text_inputs = self.tokenizer(query_text_list)
                    query_text_inputs = {
                        'input_ids': self.collate_tokens([torch.tensor(_) for _ in query_text_inputs['input_ids']],
                                                         pad_idx=0).to(
                            'cuda' if self.args.cuda else 'cpu'),
                        'attention_mask': self.collate_tokens(
                            [torch.tensor(_) for _ in query_text_inputs['attention_mask']],
                            pad_idx=0).to('cuda' if self.args.cuda else 'cpu')
                    }
                    # lhs_ = self.know_injector(
                    #     input_pattern='query',
                    #     input_embeddings=lhs_, text=query_text_inputs
                    # )
                    # rel_ = self.know_injector(
                    #     input_pattern='query',
                    #     input_embeddings=rel_, text=query_text_inputs
                    # )
                    # rhs_emb = self.know_injector(
                    #     input_pattern='query',
                    #     input_embeddings=rhs_emb, text=query_text_inputs
                    # )
                    res, _ = self.score_o(lhs_, rel_, rhs_, query_text_inputs)
                    if self.do_sigmoid is True:
                        res = torch.sigmoid(res)
                    if self.do_normalize is True:
                        res = normalize(res)
                    return res

                if graph_type == "1p":
                    scores = d2.query_1p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function)
                elif graph_type == "2p":
                    scores = d2.query_2p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "3p":
                    scores = d2.query_3p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "2i":
                    scores = d2.query_2i(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function, t_norm=t_norm)
                elif graph_type == "3i":
                    scores = d2.query_3i(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function, t_norm=t_norm)
                elif graph_type == "pi":
                    scores = d2.query_pi(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "ip":
                    scores = d2.query_ip(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "2u-DNF":
                    scores = d2.query_2u_dnf(entity_embeddings=self.embeddings[0],
                                             predicate_embeddings=self.embeddings[1],
                                             queries=queries,
                                             scoring_function=scoring_function,
                                             t_conorm=t_conorm)
                elif graph_type == "up-DNF":
                    scores = d2.query_up_dnf(entity_embeddings=self.embeddings[0],
                                             predicate_embeddings=self.embeddings[1],
                                             queries=queries,
                                             scoring_function=scoring_function,
                                             k=self.k, t_norm=t_norm, t_conorm=t_conorm)
                else:
                    raise ValueError(f'Unknown query type: {graph_type}')

        return None, scores, None, all_idxs
