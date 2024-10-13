# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Tuple, List, Dict
from collections import defaultdict
import torch
from torch import nn
import pickle
import os
from collections import OrderedDict
import numpy as np
from transformers import AutoConfig, AutoTokenizer
# from kbc.src.qto_know_inject import KnowInjector
from qto_know_inject import KnowInjector

def filtering(scores, these_queries, filters, n_rel, n_ent, 
              c_begin, chunk_size, query_type):
    # set filtered and true scores to -1e6 to be ignored
    # take care that scores are chunked
    for i, query in enumerate(these_queries):
        existing_s = (query[0].item(), query[1].item()) in filters # reciprocal training always has candidates = rhs
        existing_r = (query[2].item(), query[1].item() + n_rel) in filters # standard training separate rhs and lhs
        if query_type == 'rhs':
            if existing_s:
                filter_out = filters[(query[0].item(), query[1].item())]
                # filter_out += [queries[b_begin + i, 2].item()]
                filter_out += [query[2].item()]
        if query_type == 'lhs':
            if existing_r:
                filter_out = filters[(query[2].item(), query[1].item() + n_rel)]
                # filter_out += [queries[b_begin + i, 0].item()]    
                filter_out += [query[0].item()]                         
        if query_type == 'rel':
            pass
        if chunk_size < n_ent:
            filter_in_chunk = [
                    int(x - c_begin) for x in filter_out
                    if c_begin <= x < c_begin + chunk_size
            ]
            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
        else:
            scores[i, torch.LongTensor(filter_out)] = -1e6
    return scores

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

class KBCModel(nn.Module):
    def get_candidates(self, chunk_begin, chunk_size, target='rhs', indices=None):
        """
        Get scoring candidates for (q, ?)
        """
        pass

    def get_queries(self, queries, target='rhs'):
        """
        Get queries in a comfortable format for evaluation on GPU
        """
        pass

    def score(self, x: torch.Tensor):
        pass
    
    def forward_bpr(self, pos, neg):
        pos_scores = self.score(pos)
        neg_scores = self.score(neg)
        delta = pos_scores - neg_scores
        fac = self.get_factor(torch.cat((pos, neg), dim=0))
        return delta, fac
    
    def forward_mr(self, pos, neg):
        pass

    def checkpoint(self, model_cache_path, epoch_id):
        if model_cache_path is not None:
            print('Save the model at epoch {}'.format(epoch_id))
            torch.save(self.state_dict(), model_cache_path + '{}.model'.format(epoch_id))
        
    def get_ranking(self, 
                    queries: torch.Tensor,
                    filters: Dict[Tuple[int, int], List[int]],
                    batch_size: int = 1000, chunk_size: int = -1,
                    candidates='rhs'): 
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of answering candidates processed at once
        :return:
        """
        query_type = candidates
        if chunk_size < 0: # not chunking, score against all candidates at once
            chunk_size = self.sizes[2] # entity ranking
        ranks = torch.ones(len(queries))
        predicted = torch.zeros(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                cands = self.get_candidates(c_begin, chunk_size, target=query_type)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries, target=query_type)
                    scores = q @ cands # torch.mv MIPS
                    targets = self.score(these_queries)
                    if filters is not None:
                        scores = filtering(scores, these_queries, filters, 
                                           n_rel=self.sizes[1], n_ent=self.sizes[2], 
                                           c_begin=c_begin, chunk_size=chunk_size,
                                           query_type=query_type)
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    predicted[b_begin:b_begin + batch_size] = torch.max(scores, dim=1)[1].cpu()
                    b_begin += batch_size
                c_begin += chunk_size
        return ranks, predicted

    def get_metric_ogb(self, 
                       queries: torch.Tensor,
                       batch_size: int = 1000, 
                       query_type='rhs',
                       evaluator=None): 
        """No need to filter since the provided negatives are ready filtered
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        test_logs = defaultdict(list)
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                ##### hard code neg_indice TODO
                if these_queries.shape[1] > 5: # more than h,r,t,h_type,t_type
                    tot_neg = 1000 if evaluator.name in ['ogbl-biokg', 'ogbl-wikikg2'] else 0
                    neg_indices = these_queries[:, 3:3+tot_neg]
                    chunk_begin, chunk_size = None, None
                else:
                    neg_indices = None
                    chunk_begin, chunk_size = 0, self.sizes[2] # all the entities
                q = self.get_queries(these_queries, target=query_type)
                cands = self.get_candidates(chunk_begin, chunk_size,
                                            target=query_type,
                                            indices=neg_indices)
                if cands.dim() >= 3:# each example has a different negative candidate embedding matrix
                    scores = torch.bmm(cands, q.unsqueeze(-1)).squeeze(-1)
                else:
                    scores = q @ cands # torch.mv MIPS, pos + neg scores
                targets = self.score(these_queries) # positive scores
                batch_results = evaluator.eval({'y_pred_pos': targets.squeeze(-1), 
                                                'y_pred_neg': scores})
                del targets, scores, q, cands
                for metric in batch_results:
                    test_logs[metric].append(batch_results[metric])
                b_begin += batch_size
        metrics = {}
        for metric in test_logs:
            metrics[metric] = torch.cat(test_logs[metric]).mean().item()
        return metrics    


class TransE(KBCModel):
    def __init__(self, sizes, rank, init_size):
        super(TransE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.entity = nn.Embedding(sizes[0], rank, sparse=False)
        self.relation = nn.Embedding(sizes[1], rank, sparse=False)

        self.entity.weight.data *= init_size
        self.relation.weight.data *= init_size

    def param(self):
        return (self.entity.weight.data.detach(), self.relation.weight.data.detach())
    
    def param_grad(self):
        return (self.entity.weight.grad.data.detach(), self.relation.weight.grad.data.detach())

    def score(self, x):
        lhs = self.entity(x[:, 0])
        rel = self.relation(x[:, 1])
        rhs = self.entity(x[:, 2])
        proj = lhs + rel
        tmp1 = torch.sum(2 * proj * rhs, dim=1, keepdim=True)
        tmp2 = torch.sum(proj * proj, dim=1, keepdim=True)
        tmp3 = torch.sum(rhs * rhs, dim=1, keepdim=True)
        scores = tmp1 - tmp2 - tmp3
        return scores
   
    def forward(self, x, score_rhs=True, score_rel=False, score_lhs=False):
        lhs = self.entity(x[:, 0])
        rel = self.relation(x[:, 1])
        rhs = self.entity(x[:, 2])

        if score_rhs:
            lhs_proj = lhs + rel 
            # compute - (lhs_proj - rhs) ** 2 = 2 lhs_proj * rhs - rhs ** 2 - lhs_proj ** 2
            # tmp1 = 2 * lhs_proj @ self.entity.weight.t()
            # tmp2 = torch.norm(lhs_proj, dim=1, p=2).unsqueeze(1)
            # tmp3 = torch.norm(self.entity.weight, dim=1, p=2).unsqueeze(0)
            # rhs_scores = tmp1 - tmp2 - tmp3
            rhs_scores = (2 * lhs_proj @ self.entity.weight.t()
                          - torch.sum(lhs_proj * lhs_proj, dim=1).unsqueeze(1)
                          - torch.sum(self.entity.weight * self.entity.weight, dim=1).unsqueeze(0))

        if score_lhs:
            rhs_proj = rel - rhs
            # compute - (lhs + rhs_proj) ** 2 = -2 rhs_proj * lhs - lhs ** 2 - rhs_proj ** 2
            # tmp1 = -2 * rhs_proj @ self.entity.weight.t()
            # tmp2 = torch.norm(rhs_proj, dim=1, p=2).unsqueeze(1)
            # tmp3 = torch.norm(self.entity.weight, dim=1, p=2).unsqueeze(0)
            lhs_scores = (-2 * rhs_proj @ self.entity.weight.t()
                          - torch.sum(rhs_proj * rhs_proj, dim=1).unsqueeze(1)
                          - torch.sum(self.entity.weight * self.entity.weight, dim=1).unsqueeze(0))
    
        if score_rel:
            lr_proj = lhs - rhs
            # compute - (lr_proj + rel) ** 2 = -2 lr_proj * rel - rel ** 2 - lr_proj ** 2
            # tmp1 = -2 * lr_proj @ self.relation.weight.t()
            # tmp2 = torch.norm(lr_proj, dim=1, p=2).unsqueeze(1)
            # tmp3 = torch.norm(self.relation.weight, dim=1, p=2).unsqueeze(0)
            # rel_scores = tmp1 - tmp2 -tmp3
            rel_scores = (-2 * lr_proj @ self.relation.weight.t()
                          - torch.sum(lr_proj * lr_proj, dim=1).unsqueeze(1)
                          - torch.sum(self.relation.weight * self.relation.weight, dim=1).unsqueeze(0))

        factors = (lhs, rel, rhs)
        if score_rhs and score_rel and score_lhs:
            return (rhs_scores, rel_scores, lhs_scores), factors
        elif score_rhs and score_rel:
            return (rhs_scores, rel_scores), factors
        elif score_lhs and score_rel:
            pass
        elif score_rhs and score_lhs:
            pass
        elif score_rhs:
            return rhs_scores, factors
        elif score_rel:
            return rel_scores, factors
        elif score_lhs:
            return lhs_scores, factors
        else:
            return None

    def get_candidates(self, chunk_begin, chunk_size, target='rhs'):
        if target in ['rhs', 'lhs']:
            cands = self.entity.weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)
        elif target == 'rel':
            cands = self.relation.weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)
        else:
            cands = None
        return cands

    def get_queries(self, queries, target='rhs'):
        lhs = self.entity(queries[:, 0]).data
        rel = self.relation(queries[:, 1]).data
        rhs = self.entity(queries[:, 2]).data
        if target == 'rhs':
            queries = lhs + rel
        elif target == 'lhs':
            queries = -1 * (rel - rhs)
        if target == 'rel':
            queries = -1 * (lhs - rhs)
        return queries
              
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, 
            chunk_size: int = -1, candidates='rhs'
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of answering candidates processed at once
        :return:
        """
        if chunk_size < 0:
            if candidates in ['rhs', 'lhs']:
                chunk_size = self.sizes[2]
            else:
                chunk_size = self.sizes[1]
        ranks = torch.ones(len(queries))
        predicted = torch.zeros(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                cands = self.get_candidates(c_begin, chunk_size, target=candidates)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries, target=candidates)
                    tmp1 = 2 * q @ cands
                    tmp2 = torch.sum(q * q, dim=1).unsqueeze(1)
                    tmp3 = torch.sum(cands.transpose(0, 1) * cands.transpose(0, 1), dim=1).unsqueeze(0)
                    scores = tmp1 - tmp2 - tmp3
                    targets = self.score(these_queries)
                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    # refer to process_datasets.py + datasets.py
                    # 1) reciprocal version, all candidates are on rhs, and can be inspected using the to_skip[missing]
                    # 2) standard version, rhs is the same as reciprocal but lhs need to consider (rhs, rel + n_pred) in to_skip['lhs']
                    for i, query in enumerate(these_queries):
                        existing_s = (query[0].item(), query[1].item()) in filters # reciprocal training always has candidates = rhs
                        existing_r = (query[2].item(), query[1].item() + self.sizes[1]) in filters # standard training separate rhs and lhs
                        if candidates == 'rhs':
                            if existing_s:
                                filter_out = filters[(query[0].item(), query[1].item())]
                                filter_out += [queries[b_begin + i, 2].item()]
                        if candidates == 'lhs':
                            if existing_r:
                                filter_out = filters[(query[2].item(), query[1].item() + self.sizes[1])]
                                filter_out += [queries[b_begin + i, 0].item()]                             
                        if candidates == 'rel':
                            pass
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                    int(x - c_begin) for x in filter_out
                                    if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    predicted[b_begin:b_begin + batch_size] = torch.max(scores, dim=1)[1].cpu()
                    b_begin += batch_size
                c_begin += chunk_size
        return ranks, predicted
    
    def get_factor(self, x):
        lhs = self.entity(x[:, 0])
        rel = self.relation(x[:, 1])
        rhs = self.entity(x[:, 2])
        factors = (lhs, rel, rhs)
        return factors
   
           
class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3, opt=None
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.opt = opt

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=False)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.entity2text, self.relation2text = self.get_ent_rel_text()
        self.tokenizer = AutoTokenizer.from_pretrained(self.opt['plm_name'])
        self.know_injector = KnowInjector(args=self.opt)
        # print()

    def get_ent_rel_text(self):
        # data_path = 'kbc/src/src_data/{}'.format(self.opt['dataset'])
        data_path = 'src_data/{}'.format(self.opt['dataset'])

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

    def param(self):
        return (self.embeddings[0].weight.data.detach(), self.embeddings[1].weight.data.detach())

    def param_grad(self):
        return (self.embeddings[0].weight.grad.data.detach(), self.embeddings[1].weight.grad.data.detach())

    def score(self, x):
        if self.opt['knowledge_inject']:
            text_lhs = self.entity2text[x[:, 0].cpu()].tolist()
            text_rel = self.relation2text[x[:, 1].cpu()].tolist()
            text_rhs = self.entity2text[x[:, 2].cpu()].tolist()

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        pre_rhs_1 = lhs[0] * rel[0] - lhs[1] * rel[1]
        pre_rhs_2 = lhs[0] * rel[1] + lhs[1] * rel[0]

        if self.opt['knowledge_inject']:
            query_text_list = ['({}, ({}), ?)'.format(text_lhs[i][0], text_rel[i][0]) for i in range(len(text_lhs))]
            pre_rhs_1 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rhs_1)
            pre_rhs_2 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rhs_2)

        return torch.sum(
            pre_rhs_1 * rhs[0] +
            pre_rhs_2 * rhs[1],
            1, keepdim=True
        )

    def knowledge_inject_module(self, query_text_list, input_embeddings):
        query_text_inputs = self.tokenizer(query_text_list,
                                           max_length=self.opt['max_seq_len'],
                                           add_special_tokens=False)
        query_text_inputs = {
            'input_ids': collate_tokens([torch.tensor(_) for _ in query_text_inputs['input_ids']], pad_idx=0).to(self.opt['device']),
            'attention_mask': collate_tokens([torch.tensor(_) for _ in query_text_inputs['attention_mask']], pad_idx=0).to(self.opt['device'])
        }
        know_inject_query_embeddings = self.know_injector(
            input_pattern='query',
            input_embeddings=input_embeddings, text=query_text_inputs
        ) + input_embeddings
        return know_inject_query_embeddings

    def forward(self, x, score_rhs=True, score_rel=False, score_lhs=False):
        if self.opt['knowledge_inject']:
            text_lhs = self.entity2text[x[:, 0].cpu()].tolist()
            text_rel = self.relation2text[x[:, 1].cpu()].tolist()
            text_rhs = self.entity2text[x[:, 2].cpu()].tolist()

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        rhs_scores, rel_scores = None, None
        if score_rhs:
            pre_rhs_1 = lhs[0] * rel[0] - lhs[1] * rel[1]
            pre_rhs_2 = lhs[0] * rel[1] + lhs[1] * rel[0]

            if self.opt['knowledge_inject']:
                query_text_list = ['({}, ({}), ?)'.format(text_lhs[i][0], text_rel[i][0]) for i in range(len(text_lhs))]
                pre_rhs_1 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rhs_1)
                pre_rhs_2 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rhs_2)

            to_score_entity = self.embeddings[0].weight
            to_score_entity = to_score_entity[:, :self.rank], to_score_entity[:, self.rank:]
            rhs_scores = (
                pre_rhs_1 @ to_score_entity[0].transpose(0, 1) +
                pre_rhs_2 @ to_score_entity[1].transpose(0, 1)
            )
        if score_rel:
            pre_rel_1 = lhs[0] * rhs[0] + lhs[1] * rhs[1]
            pre_rel_2 = lhs[0] * rhs[1] - lhs[1] * rhs[0]

            if self.opt['knowledge_inject']:
                query_text_list = ['({}, (?), {})'.format(text_lhs[i][0], text_rhs[i][0]) for i in range(len(text_lhs))]
                # query_text_list = ['({}, (), {})'.format(text_lhs[i][0], text_rhs[i][0]) for i in range(len(text_lhs))]
                pre_rel_1 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rel_1)
                pre_rel_2 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rel_2)

            to_score_rel = self.embeddings[1].weight
            to_score_rel = to_score_rel[:, :self.rank], to_score_rel[:, self.rank:]
            rel_scores = (
                pre_rel_1 @ to_score_rel[0].transpose(0, 1) +
                pre_rel_2 @ to_score_rel[1].transpose(0, 1)
            )
        if score_lhs:
            pre_lhs_1 = rel[0] * rhs[0] + rel[1] * rhs[1]
            pre_lhs_2 = rel[0] * rhs[1] - rel[1] * rhs[0]

            if self.opt['knowledge_inject']:
                query_text_list = ['(?, ({}), {})'.format(text_rel[i][0], text_rhs[i][0]) for i in range(len(text_lhs))]
                # query_text_list = ['(, ({}), {})'.format(text_rel[i][0], text_rhs[i][0]) for i in range(len(text_lhs))]
                pre_lhs_1 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_lhs_1)
                pre_lhs_2 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_lhs_2)

            to_score_lhs = self.embeddings[0].weight
            to_score_lhs = to_score_lhs[:, :self.rank], to_score_lhs[:, self.rank:]
            lhs_scores = (
                pre_lhs_1 @ to_score_lhs[0].transpose(0, 1) +
                pre_lhs_2 @ to_score_lhs[1].transpose(0, 1)
            )

        factors = self.get_factor(x)
        if score_rhs and score_rel and score_lhs:
            return (rhs_scores, rel_scores, lhs_scores), factors
        elif score_rhs and score_rel:
            return (rhs_scores, rel_scores), factors
        elif score_lhs and score_rel:
            pass
        elif score_rhs and score_lhs:
            return (rhs_scores, lhs_scores), factors
        elif score_rhs:
            return rhs_scores, factors
        elif score_rel:
            return rel_scores, factors
        elif score_lhs:
            return lhs_scores, factors
        else:
            return None

    def get_candidates(self, chunk_begin=None, chunk_size=None, target='rhs', indices=None):
        if target == 'rhs' or target == 'lhs': #TODO: extend to other models
            if indices == None:
                return self.embeddings[0].weight.data[
                    chunk_begin:chunk_begin + chunk_size
                ].transpose(0, 1)
            else:
                bsz = indices.shape[0]
                num_cands = indices.shape[1]
                if target == 'rhs':
                    indices = indices[:, num_cands//2:]
                else:
                    indices = indices[:, 0:num_cands//2]
                return self.embeddings[0].weight.data[indices.reshape(-1)].reshape(bsz, num_cands//2, -1)
        elif target == 'rel':
            return self.embeddings[1].weight.data[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)
        
    def get_queries(self, queries, target='rhs'):
        if self.opt['knowledge_inject']:
            text_lhs = self.entity2text[queries[:, 0].cpu()].tolist()
            text_rel = self.relation2text[queries[:, 1].cpu()].tolist()
            text_rhs = self.entity2text[queries[:, 2].cpu()].tolist()

        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rhs = self.embeddings[0](queries[:, 2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        if target == 'rhs':
            pre_rhs_1 = lhs[0] * rel[0] - lhs[1] * rel[1]
            pre_rhs_2 = lhs[0] * rel[1] + lhs[1] * rel[0]

            if self.opt['knowledge_inject']:
                query_text_list = ['({}, ({}), ?)'.format(text_lhs[i][0], text_rel[i][0]) for i in range(len(text_lhs))]
                pre_rhs_1 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rhs_1)
                pre_rhs_2 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rhs_2)

            # return torch.cat([
            #     lhs[0] * rel[0] - lhs[1] * rel[1],
            #     lhs[0] * rel[1] + lhs[1] * rel[0]
            # ], 1)
            return torch.cat([
                pre_rhs_1,
                pre_rhs_2
            ], 1)
        elif target == 'lhs':
            pre_lhs_1 = rel[0] * rhs[0] + rel[1] * rhs[1]
            pre_lhs_2 = rel[0] * rhs[1] - rel[1] * rhs[0]

            if self.opt['knowledge_inject']:
                query_text_list = ['(?, ({}), {})'.format(text_rel[i][0], text_rhs[i][0]) for i in range(len(text_lhs))]
                # query_text_list = ['(, ({}), {})'.format(text_rel[i][0], text_rhs[i][0]) for i in range(len(text_lhs))]
                pre_lhs_1 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_lhs_1)
                pre_lhs_2 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_lhs_2)
            return torch.cat([
                pre_lhs_1,
                pre_lhs_2
            ], 1)
        elif target == 'rel':
            pre_rel_1 = lhs[0] * rhs[0] + lhs[1] * rhs[1]
            pre_rel_2 = lhs[0] * rhs[1] - lhs[1] * rhs[0]

            if self.opt['knowledge_inject']:
                query_text_list = ['({}, (?), {})'.format(text_lhs[i][0], text_rhs[i][0]) for i in range(len(text_lhs))]
                # query_text_list = ['({}, (), {})'.format(text_lhs[i][0], text_rhs[i][0]) for i in range(len(text_lhs))]
                pre_rel_1 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rel_1)
                pre_rel_2 = self.knowledge_inject_module(query_text_list=query_text_list, input_embeddings=pre_rel_2)

            return torch.cat([
                pre_rel_1,
                pre_rel_2
            ], 1)

    def get_factor(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        return (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))

class TuckER(KBCModel):
    def __init__(self, sizes, rank_e, rank_r, init_size=1e-3, dp=0.5):
        super(TuckER, self).__init__()
        self.sizes = sizes
        self.rank_e = rank_e
        self.rank_r = rank_r
        self.core = nn.Parameter(torch.rand(rank_e, rank_r, rank_e) * init_size)
        self.entity = nn.Embedding(sizes[0], rank_e, sparse=True)
        self.relation = nn.Embedding(sizes[1], rank_r, sparse=True)
        self.dropout = torch.nn.Dropout(dp)

        self.entity.weight.data *= init_size
        self.relation.weight.data *= init_size       
    
    def score(self, x):
        lhs = self.entity(x[:, 0])
        rel = self.relation(x[:, 1])
        rhs = self.entity(x[:, 2])

        lhs_proj = torch.matmul(self.core.transpose(0, 2), lhs.transpose(0, 1)).transpose(0, 2) # b, rank_r, rank_e
        rel_proj = rel.view(-1, 1, self.rank_r)
        lhs_proj = torch.bmm(rel_proj, lhs_proj).view(-1, self.rank_e)
        return torch.sum(lhs_proj * rhs, 1, keepdim=True)

    def forward(self, x, score_rhs=True, score_rel=False, score_lhs=False, normalize_rel=False):
        lhs = self.entity(x[:, 0])
        rel = self.relation(x[:, 1])
        rhs = self.entity(x[:, 2]) 

        if score_rhs:
            lhs_proj = torch.matmul(self.core.transpose(0, 2), lhs.transpose(0, 1)).transpose(0, 2) # b, rank_r, rank_e
            rel_proj = rel.view(-1, 1, self.rank_r)
            lhs_proj = torch.bmm(rel_proj, 
                                 self.dropout(lhs_proj)).view(-1, self.rank_e)
            rhs_scores = lhs_proj @ self.entity.weight.t()
        if score_rel:
            lhs_proj = torch.matmul(self.core.transpose(0, 2), lhs.transpose(0, 1)).transpose(0, 2) # b, rank_r, rank_e
            rhs_proj = rhs.view(-1, self.rank_e, 1)
            lr_proj = torch.bmm(self.dropout(lhs_proj), 
                                rhs_proj).view(-1, self.rank_r) # b, rank_r
            rel_scores = lr_proj @ self.relation.weight.t()
        if score_lhs:
            rhs_proj = torch.matmul(self.core, rhs.transpose(0, 1)).transpose(0, 2) # b, rank_r, rank_e
            rel_proj = rel.view(-1, 1, self.rank_r)
            rhs_proj = torch.bmm(rel_proj, 
                                 self.dropout(rhs_proj)).view(-1, self.rank_e)
            lhs_scores = rhs_proj @ self.entity.weight.t()

        factors = (lhs, 
                   rel * ((self.rank_e * 1.0 / self.rank_r) ** (1/3.0)), 
                   rhs) # the rank of relation is smaller than that of entity, so we add some scaling
        if score_rhs and score_rel and score_lhs:
            return (rhs_scores, rel_scores, lhs_scores), factors
        elif score_rhs and score_rel:
            return (rhs_scores, rel_scores), factors
        elif score_lhs and score_rel:
            pass
        elif score_rhs and score_lhs:
            pass
        elif score_rhs:
            return rhs_scores, factors
        elif score_rel:
            return rel_scores, factors
        elif score_lhs:
            return lhs_scores, factors
        else:
            return None
    
    def get_candidates(self, chunk_begin, chunk_size, target='rhs'):
        if target in ['rhs', 'lhs']:
            cands = self.entity.weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)
        elif target == 'rel':
            cands = self.relation.weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)
        else:
            cands = None
        return cands

    def get_queries(self, queries, target='rhs'):
        lhs = self.entity(queries[:, 0]).data
        rel = self.relation(queries[:, 1]).data
        rhs = self.entity(queries[:, 2]).data

        if target == 'rhs':
            lhs_proj = torch.matmul(self.core.data.transpose(0, 2), lhs.transpose(0, 1)).transpose(0, 2) # b, rank_r, rank_e
            rel_proj = rel.view(-1, 1, self.rank_r)
            queries = torch.bmm(rel_proj, lhs_proj).view(-1, self.rank_e)
        elif target == 'rel':
            lhs_proj = torch.matmul(self.core.data.transpose(0, 2), lhs.transpose(0, 1)).transpose(0, 2) # b, rank_r, rank_e
            rhs_proj = rhs.view(-1, self.rank_e, 1)
            queries = torch.bmm(lhs_proj, rhs_proj).view(-1, self.rank_r)
        elif target == 'lhs':
            rhs_proj = torch.matmul(self.core.data, rhs.transpose(0, 1)).transpose(0, 2) # b, rank_r, rank_e
            rel_proj = rel.view(-1, 1, self.rank_r)
            queries = torch.bmm(rel_proj, rhs_proj).view(-1, self.rank_e)
        return queries


class CP(KBCModel):
    def __init__(self, sizes, rank, init_size):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=False)
        self.rel = nn.Embedding(sizes[1], rank, sparse=False)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=False)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size
    
    def param(self):
        return (self.lhs.weight.data.detach(), self.rel.weight.data.detach(), self.rhs.weight.data.detach())
    
    def param_grad(self):
        return (self.lhs.weight.grad.data.detach(), self.rel.weight.grad.data.detach(), self.rhs.weight.grad.data.detach())

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x, score_rhs=True, score_rel=False, score_lhs=False, normalize_rel=False):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        rhs_scores, rel_scores = None, None
        if score_rhs:
            rhs_scores = (lhs * rel) @ self.rhs.weight.t()
        if score_rel:
            rel_scores = (lhs * rhs) @ self.rel.weight.t()
        if score_lhs:
            lhs_scores = (rhs * rel) @ self.lhs.weight.t()

        factors = self.get_factor(x)
        if score_rhs and score_rel and score_lhs:
            return (rhs_scores, rel_scores, lhs_scores), factors
        elif score_rhs and score_rel:
            return (rhs_scores, rel_scores), factors
        elif score_lhs and score_rel:
            pass
        elif score_rhs and score_lhs:
            pass
        elif score_rhs:
            return rhs_scores, factors
        elif score_rel:
            return rel_scores, factors
        elif score_lhs:
            return lhs_scores, factors
        else:
            return None

    def get_candidates(self, chunk_begin, chunk_size, target='rhs'):
        if target == 'rhs':
            return self.rhs.weight.data[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)
        elif target == 'lhs':
            return self.lhs.weight.data[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)
        elif target == 'rel':
            return self.rel.weight.data[
                chunk_begin:chunk_begin + chunk_size
            ].transpose(0, 1)

    def get_queries(self, queries, target='rhs'):
        if target == 'rhs':
            return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data
        elif target == 'lhs':
            return self.rhs(queries[:, 2]).data * self.rel(queries[:, 1]).data
        elif target == 'rel':
            return self.lhs(queries[:, 0]).data * self.rhs(queries[:, 2]).data

    def get_factor(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        factors = (lhs, rel, rhs)
        return factors


class RESCAL(KBCModel):
    def __init__(
        self, sizes, rank, init_size=1e-3
    ):
        super(RESCAL, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.entity = nn.Embedding(sizes[0], rank, sparse=False)
        self.relation = nn.Embedding(sizes[1], rank * rank, sparse=False)
        
        self.entity.weight.data *= init_size
        self.relation.weight.data *= init_size
    
    def score(self, x):
        """Note: should make sure this score is the same as q @ cands"""
        lhs = self.entity(x[:, 0])
        rel = self.relation(x[:, 1])
        rhs = self.entity(x[:, 2])
        rel = rel.view(-1, self.rank, self.rank)
        lhs_proj = lhs.view(-1, 1, self.rank)
        lhs_proj = torch.bmm(lhs_proj, rel).view(-1, self.rank)
        return torch.sum(lhs_proj * rhs, 1, keepdim=True)

    def forward(self, x, score_rhs=True, score_rel=False, score_lhs=False, normalize_rel=False):
        lhs = self.entity(x[:, 0])
        rel = self.relation(x[:, 1])
        rhs = self.entity(x[:, 2])

        rel = rel.view(-1, self.rank, self.rank)
        
        if score_rhs:
            lhs_proj = lhs.view(-1, 1, self.rank)
            lhs_proj = torch.bmm(lhs_proj, rel).view(-1, self.rank)
            rhs_scores = lhs_proj @ self.entity.weight.t()
        if score_rel:
            lhs_proj = lhs.view(-1, self.rank, 1)
            rhs_proj = rhs.view(-1, 1, self.rank)
            lr_proj = torch.bmm(lhs_proj, rhs_proj).view(-1, self.rank * self.rank)
            rel_scores = lr_proj @ self.relation.weight.t()
        if score_lhs:
            rhs_proj = rhs.view(-1, 1, self.rank)
            rhs_proj = torch.bmm(rhs_proj, rel.transpose(1, 2)).view(-1, self.rank)
            lhs_scores = rhs_proj @ self.entity.weight.t()

        # factors = (lhs, rel, rhs) if not normalize_rel else 
        factors = (lhs, rel / (self.rank ** (1/3.0)), rhs) # scaling factor for N3
        if score_rhs and score_rel and score_lhs:
            return (rhs_scores, rel_scores, lhs_scores), factors
        elif score_rhs and score_rel:
            return (rhs_scores, rel_scores), factors
        elif score_lhs and score_rel:
            pass
        elif score_rhs and score_lhs:
            return (rhs_scores, lhs_scores), factors
        elif score_rhs:
            return rhs_scores, factors
        elif score_rel:
            return rel_scores, factors
        elif score_lhs:
            return lhs_scores, factors
        else:
            return None

    def get_candidates(self, chunk_begin, chunk_size, target='rhs'):
        if target in ['rhs', 'lhs']:
            cands = self.entity.weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)
        elif target == 'rel':
            cands = self.relation.weight.data[chunk_begin:chunk_begin + chunk_size].transpose(0, 1)
        else:
            cands = None
        return cands

    def get_queries(self, queries, target='rhs'):
        lhs = self.entity(queries[:, 0]).data
        rel = self.relation(queries[:, 1]).data
        rhs = self.entity(queries[:, 2]).data
        rel = rel.view(-1, self.rank, self.rank)
        if target == 'rhs':
            lhs_proj = lhs.view(-1, 1, self.rank)
            queries = torch.bmm(lhs_proj, rel).view(-1, self.rank)
        elif target == 'rel':
            lhs_proj = lhs.view(-1, self.rank, 1)
            rhs_proj = rhs.view(-1, 1, self.rank)
            queries = torch.bmm(lhs_proj, rhs_proj).view(-1, self.rank * self.rank)
        elif target == 'lhs':
            rhs_proj = rhs.view(-1, 1, self.rank)
            queries = torch.bmm(rhs_proj, rel.transpose(1, 2)).view(-1, self.rank)
        return queries



