#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import Dataset
from util import list2tuple, tuple2list, flatten, collate_tokens


class TestDataset(Dataset):
    def __init__(self, queries, query2text, entity2text_inputs, entity2text, relation2text, nentity, nrelation, args):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.query2text = query2text
        self.entity2text_inputs = entity2text_inputs
        self.entity2text = entity2text
        self.relation2text = relation2text
        self.nentity = nentity
        self.nrelation = nrelation
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
        self.args = args

        self.query_name_dict = {('e', ('r',)): '1p',
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

    def __len__(self):
        return self.len

    def get_query_text(self, query, query_structure):
        if self.query_name_dict[query_structure] == '1p':
            return '({}, ({}))'.format(self.entity2text[query[0]], self.relation2text[query[1]])
        elif self.query_name_dict[query_structure] == '2p':
            return '({}, ({}, {}))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                           self.relation2text[query[2]])
        elif self.query_name_dict[query_structure] == '3p':
            return '({}, ({}, {}, {}))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                               self.relation2text[query[2]], self.relation2text[query[3]])
        elif self.query_name_dict[query_structure] == '2i':
            return '(({}, ({})), ({}, ({})))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                                     self.entity2text[query[2]], self.relation2text[query[3]])
        elif self.query_name_dict[query_structure] == '3i':
            return '(({}, ({})), ({}, ({})), ({}, ({})))'.format(self.entity2text[query[0]],
                                                                 self.relation2text[query[1]],
                                                                 self.entity2text[query[2]],
                                                                 self.relation2text[query[3]],
                                                                 self.entity2text[query[4]],
                                                                 self.relation2text[query[5]])
        elif self.query_name_dict[query_structure] == 'ip':
            return '((({}, ({})), ({}, ({}))), ({}))'.format(self.entity2text[query[0]],
                                                             self.relation2text[query[1]],
                                                             self.entity2text[query[2]],
                                                             self.relation2text[query[3]],
                                                             self.relation2text[query[4]])
        elif self.query_name_dict[query_structure] == 'pi':
            return '(({}, ({}, {})), ({}, ({})))'.format(self.entity2text[query[0]],
                                                         self.relation2text[query[1]],
                                                         self.relation2text[query[2]],
                                                         self.entity2text[query[3]],
                                                         self.relation2text[query[4]])
        elif self.query_name_dict[query_structure] == '2in':
            return '(({}, ({})), ({}, ({}, negative)))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                                               self.entity2text[query[2]], self.relation2text[query[3]])
        elif self.query_name_dict[query_structure] == '3in':
            return '(({}, ({})), ({}, ({})), ({}, ({}, negative)))'.format(self.entity2text[query[0]],
                                                                           self.relation2text[query[1]],
                                                                           self.entity2text[query[2]],
                                                                           self.relation2text[query[3]],
                                                                           self.entity2text[query[4]],
                                                                           self.relation2text[query[5]])
        elif self.query_name_dict[query_structure] == 'inp':
            return '((({}, ({})), ({}, ({}, negative))), ({}))'.format(self.entity2text[query[0]],
                                                                       self.relation2text[query[1]],
                                                                       self.entity2text[query[2]],
                                                                       self.relation2text[query[3]],
                                                                       self.relation2text[query[5]])
        elif self.query_name_dict[query_structure] == 'pin':
            return '(({}, ({}, {})), ({}, ({}, negative)))'.format(self.entity2text[query[0]],
                                                                   self.relation2text[query[1]],
                                                                   self.relation2text[query[2]],
                                                                   self.entity2text[query[3]],
                                                                   self.relation2text[query[4]])
        elif self.query_name_dict[query_structure] == 'pni':
            return '(({}, ({}, {}, negative)), ({}, ({})))'.format(self.entity2text[query[0]],
                                                                   self.relation2text[query[1]],
                                                                   self.relation2text[query[2]],
                                                                   self.entity2text[query[4]],
                                                                   self.relation2text[query[5]])
        elif self.query_name_dict[query_structure] == '2u-DNF':
            return ['({}, ({}))'.format(self.entity2text[query[0]], self.relation2text[query[1]]),
                    '({}, ({}))'.format(self.entity2text[query[2]], self.relation2text[query[3]])]
            # return '(({}, ({})), ({}, ({},)), (union))'.format(self.entity2text[query[0]],
            #                                                    self.relation2text[query[1]],
            #                                                    self.entity2text[query[2]],
            #                                                    self.relation2text[query[3]])
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ['({}, ({}, {}))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                            self.relation2text[query[5]]),
                    '({}, ({}, {}))'.format(self.entity2text[query[2]], self.relation2text[query[3]],
                                            self.relation2text[query[5]])]
            # return '((({}, ({})), ({}, ({})), (union)), ({}))'.format(self.entity2text[query[0]],
            #                                                           self.relation2text[query[1]],
            #                                                           self.entity2text[query[2]],
            #                                                           self.relation2text[query[3]],
            #                                                           self.relation2text[query[5]])

    def __getitem__(self, idx):
        # if idx == 5:
        #     print()
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        negative_sample = torch.LongTensor(range(self.nentity))

        if query_structure in [(('e', ('r',)), ('e', ('r',)), ('u',)),
                               ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))]:
            # query_text_1, query_text_2 = self.get_query_text(flatten(query), query_structure)
            # query_text_inputs_1 = self.tokenizer(query_text_1, max_length=self.args.max_seq_len)
            # query_text_inputs_2 = self.tokenizer(query_text_2, max_length=self.args.max_seq_len)
            query_text_inputs_1 = self.query2text[query]['query_text_inputs_1']
            query_text_inputs_2 = self.query2text[query]['query_text_inputs_2']
            return_dict = {'negative_sample': negative_sample,
                           'flatten_query': flatten(query),
                           'queries_unflatten': query,
                           'query_structure': query_structure,
                           'query_text_inputs_1': query_text_inputs_1,
                           'query_text_inputs_2': query_text_inputs_2}
        else:
            # query_text = self.get_query_text(flatten(query), query_structure)
            # query_text_inputs = self.tokenizer(query_text, max_length=self.args.max_seq_len)
            # if self.query_name_dict[query_structure] == 'pni':
            #     print()
            query_text_inputs = self.query2text[query]
            # print(self.tokenizer.decode(query_text_inputs['input_ids']))
            return_dict = {'negative_sample': negative_sample,
                           'flatten_query': flatten(query),
                           'queries_unflatten': query,
                           'query_structure': query_structure,
                           'query_text_inputs': query_text_inputs}
        return return_dict

        # return negative_sample, flatten(query), query, query_structure

    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_['negative_sample'] for _ in data], dim=0)
        query = [_['flatten_query'] for _ in data]
        query_unflatten = [_['queries_unflatten'] for _ in data]
        query_structure = [_['query_structure'] for _ in data]

        if query_structure[0] in [(('e', ('r',)), ('e', ('r',)), ('u',)),
                                  ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))]:
            input_ids = collate_tokens(
                [torch.tensor(_['query_text_inputs_1']['input_ids']).unsqueeze(0) for _ in data]
                +
                [torch.tensor(_['query_text_inputs_2']['input_ids']).unsqueeze(0) for _ in data],
                pad_idx=0
            )
            attention_mask = collate_tokens(
                [torch.tensor(_['query_text_inputs_1']['attention_mask']).unsqueeze(0) for _ in data]
                +
                [torch.tensor(_['query_text_inputs_2']['attention_mask']).unsqueeze(0) for _ in data],
                pad_idx=0
            )
            query_text_input = {
                'input_ids': input_ids.unsqueeze(0),
                'attention_mask': attention_mask.unsqueeze(0)
            }
        else:
            query_text_input = {
                'input_ids': collate_tokens([torch.tensor(_['query_text_inputs']['input_ids']) for _ in data],
                                            pad_idx=0),
                'attention_mask': collate_tokens([torch.tensor(_['query_text_inputs']['attention_mask']) for _ in data],
                                                 pad_idx=0)
            }
        return_dict = {'negative_sample': negative_sample,
                       'query': query,
                       'query_unflatten': query_unflatten,
                       'query_structure': query_structure,
                       'query_text': query_text_input}

        return return_dict
        # return negative_sample, query, query_unflatten, query_structure


class TrainDataset(Dataset):
    def __init__(self, queries, query2text, entity2text_inputs, entity2text, relation2text, nentity, nrelation,
                 negative_sample_size, answer, args):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.query2text = query2text
        self.entity2text_inputs = entity2text_inputs
        self.entity2text = entity2text
        self.relation2text = relation2text
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer
        # self.plm_config = AutoConfig.from_pretrained(args.plm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
        self.args = args

        self.query_name_dict = {('e', ('r',)): '1p',
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

    def __len__(self):
        return self.len

    def get_query_text(self, query, query_structure):
        if self.query_name_dict[query_structure] == '1p':
            return '({}, ({}))'.format(self.entity2text[query[0]], self.relation2text[query[1]])
        elif self.query_name_dict[query_structure] == '2p':
            return '({}, ({}, {}))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                           self.relation2text[query[2]])
        elif self.query_name_dict[query_structure] == '3p':
            return '({}, ({}, {}, {}))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                               self.relation2text[query[2]], self.relation2text[query[3]])
        elif self.query_name_dict[query_structure] == '2i':
            return '(({}, ({})), ({}, ({})))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                                     self.entity2text[query[2]], self.relation2text[query[3]])
        elif self.query_name_dict[query_structure] == '3i':
            return '(({}, ({})), ({}, ({})), ({}, ({})))'.format(self.entity2text[query[0]],
                                                                 self.relation2text[query[1]],
                                                                 self.entity2text[query[2]],
                                                                 self.relation2text[query[3]],
                                                                 self.entity2text[query[4]],
                                                                 self.relation2text[query[5]])
        elif self.query_name_dict[query_structure] == 'ip':
            return '((({}, ({})), ({}, ({}))), ({}))'.format(self.entity2text[query[0]],
                                                             self.relation2text[query[1]],
                                                             self.entity2text[query[2]],
                                                             self.relation2text[query[3]],
                                                             self.relation2text[query[4]])
        elif self.query_name_dict[query_structure] == 'pi':
            return '(({}, ({}, {})), ({}, ({})))'.format(self.entity2text[query[0]],
                                                         self.relation2text[query[1]],
                                                         self.relation2text[query[2]],
                                                         self.entity2text[query[3]],
                                                         self.relation2text[query[4]])
        elif self.query_name_dict[query_structure] == '2in':
            return '(({}, ({})), ({}, ({}, negative)))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                                               self.entity2text[query[2]], self.relation2text[query[3]])
        elif self.query_name_dict[query_structure] == '3in':
            return '(({}, ({})), ({}, ({})), ({}, ({}, negative)))'.format(self.entity2text[query[0]],
                                                                           self.relation2text[query[1]],
                                                                           self.entity2text[query[2]],
                                                                           self.relation2text[query[3]],
                                                                           self.entity2text[query[4]],
                                                                           self.relation2text[query[5]])
        elif self.query_name_dict[query_structure] == 'inp':
            return '((({}, ({})), ({}, ({}, negative))), ({}))'.format(self.entity2text[query[0]],
                                                                       self.relation2text[query[1]],
                                                                       self.entity2text[query[2]],
                                                                       self.relation2text[query[3]],
                                                                       self.relation2text[query[5]])
        elif self.query_name_dict[query_structure] == 'pin':
            return '(({}, ({}, {})), ({}, ({}, negative)))'.format(self.entity2text[query[0]],
                                                                   self.relation2text[query[1]],
                                                                   self.relation2text[query[2]],
                                                                   self.entity2text[query[3]],
                                                                   self.relation2text[query[4]])
        elif self.query_name_dict[query_structure] == 'pni':
            return '(({}, ({}, {}, negative)), ({}, ({})))'.format(self.entity2text[query[0]],
                                                                   self.relation2text[query[1]],
                                                                   self.relation2text[query[2]],
                                                                   self.entity2text[query[4]],
                                                                   self.relation2text[query[5]])
        elif self.query_name_dict[query_structure] == '2u-DNF':
            return ['({}, ({}))'.format(self.entity2text[query[0]], self.relation2text[query[1]]),
                    '({}, ({}))'.format(self.entity2text[query[2]], self.relation2text[query[3]])]
            # return '(({}, ({})), ({}, ({},)), (union))'.format(self.entity2text[query[0]],
            #                                                    self.relation2text[query[1]],
            #                                                    self.entity2text[query[2]],
            #                                                    self.relation2text[query[3]])
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ['({}, ({}, {}))'.format(self.entity2text[query[0]], self.relation2text[query[1]],
                                            self.relation2text[query[5]]),
                    '({}, ({}, {}))'.format(self.entity2text[query[2]], self.relation2text[query[3]],
                                            self.relation2text[query[5]])]
            # return '((({}, ({})), ({}, ({})), (union)), ({}))'.format(self.entity2text[query[0]],
            #                                                           self.relation2text[query[1]],
            #                                                           self.entity2text[query[2]],
            #                                                           self.relation2text[query[3]],
            #                                                           self.relation2text[query[5]])

    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.answer[query],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])

        # # query_text = self.get_query_text(flatten(query), query_structure)
        # # query_text_inputs = self.tokenizer(query_text, max_length=self.args.max_seq_len)  # , add_special_tokens=False
        query_text_inputs = self.query2text[query]

        # # positive_sample_text = self.entity2text[tail]
        # # positive_sample_text_inputs = self.tokenizer(positive_sample_text, max_length=self.args.max_seq_len)
        # positive_sample_text_inputs = self.entity2text_inputs[tail]
        #
        # negative_sample_text_inputs = []
        # for i in negative_sample:
        #     negative_sample_text_inputs.append(self.entity2text_inputs[int(i)])

        return_dict = {'positive_sample': positive_sample,
                       'negative_sample': negative_sample,
                       'subsampling_weight': subsampling_weight,
                       'flatten_query': flatten(query),
                       'query_structure': query_structure,
                       'query_text': query_text_inputs,
                       # 'positive_sample_text_inputs': positive_sample_text_inputs,
                       # 'negative_sample_text_inputs': negative_sample_text_inputs
                       }

        return return_dict

        # return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure, query_text

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_['positive_sample'] for _ in data], dim=0)
        negative_sample = torch.stack([_['negative_sample'] for _ in data], dim=0)
        subsample_weight = torch.cat([_['subsampling_weight'] for _ in data], dim=0)
        query = [_['flatten_query'] for _ in data]
        query_structure = [_['query_structure'] for _ in data]

        query_text_input = {
            'input_ids': collate_tokens([torch.tensor(_['query_text']['input_ids']) for _ in data], pad_idx=0),
            'attention_mask': collate_tokens([torch.tensor(_['query_text']['attention_mask']) for _ in data],
                                             pad_idx=0)
        }
        # positive_sample_text = {
        #     'input_ids': collate_tokens([torch.tensor(_['positive_sample_text_inputs']['input_ids']) for _ in data],
        #                                 pad_idx=0),
        #     'attention_mask': collate_tokens(
        #         [torch.tensor(_['positive_sample_text_inputs']['attention_mask']) for _ in data], pad_idx=0)
        # }
        #
        # negative_sample_text_input_ids = []
        # negative_sample_text_input_attention_mask = []
        # for l in range(len(data)):
        #     negative_sample_text_input_ids += [torch.tensor(_['input_ids']) for _ in
        #                                        data[l]['negative_sample_text_inputs']]
        #     negative_sample_text_input_attention_mask += [torch.tensor(_['attention_mask']) for _ in
        #                                                   data[l]['negative_sample_text_inputs']]
        # negative_sample_text = {
        #     'input_ids': torch.stack(
        #         torch.chunk(collate_tokens(negative_sample_text_input_ids, pad_idx=0), len(data), dim=0), dim=0),
        #     'attention_mask': torch.stack(
        #         torch.chunk(collate_tokens(negative_sample_text_input_attention_mask, pad_idx=0), len(data), dim=0),
        #         dim=0)
        # }

        return_dict = {'positive_sample': positive_sample,
                       'negative_sample': negative_sample,
                       'subsample_weight': subsample_weight,
                       'query': query,
                       'query_structure': query_structure,
                       'query_text': query_text_input,
                       # 'positive_sample_text': positive_sample_text,
                       # 'negative_sample_text': negative_sample_text
                       }
        return return_dict

    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data