#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import KGReasoning, KGFuzzyReasoning, MLP2Vec
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from tensorboardX import SummaryWriter
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
from util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple
from config import parse_args
import collections

query_name_dict = {('e', ('r',)): '1p',
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
name_answer_dict = {'1p': ['e', ['r',], 'e'],
                    '2p': ['e', ['r', 'e', 'r'], 'e'],
                    '3p': ['e', ['r', 'e', 'r', 'e', 'r'], 'e'],
                    '2i': [['e', ['r',], 'e'], ['e', ['r',], 'e'], 'e'],
                    '3i': [['e', ['r',], 'e'], ['e', ['r',], 'e'], ['e', ['r',], 'e'], 'e'],
                    'ip': [[['e', ['r',], 'e'], ['e', ['r',], 'e'], 'e'], ['r',], 'e'],
                    'pi': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r',], 'e'], 'e'],
                    '2in': [['e', ['r',], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    '3in': [['e', ['r',], 'e'], ['e', ['r',], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'inp': [[['e', ['r',], 'e'], ['e', ['r', 'n'], 'e'], 'e'], ['r',], 'e'],
                    'pin': [['e', ['r', 'e', 'r'], 'e'], ['e', ['r', 'n'], 'e'], 'e'],
                    'pni': [['e', ['r', 'e', 'r', 'n'], 'e'], ['e', ['r',], 'e'], 'e'],
                    '2u-DNF': [['e', ['r',], 'e'], ['e', ['r',], 'e'], ['u',], 'e'],
                    'up-DNF': [[['e', ['r',], 'e'], ['e', ['r',], 'e'], ['u',], 'e'], ['r',], 'e'],
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(
    name_query_dict.keys())  # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode + " " + query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]),
                              metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average' % mode, step, average_metrics)

    return all_metrics

def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")
    train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))

    ent2id = pickle.load(open(os.path.join(args.data_path, "ent2id.pkl"), 'rb'))
    rel2id = pickle.load(open(os.path.join(args.data_path, "rel2id.pkl"), 'rb'))
    entity2text_inputs = pickle.load(open(os.path.join(args.data_path, "entity2text.pkl"), 'rb'))

    entity2text = {}
    with open(os.path.join(args.data_path, "entity2text.txt"), 'r', encoding='utf-8') as fin:
        for l in fin:
            entity, text = l.strip().split('\t')
            if entity in ent2id.keys():
                entity2text[ent2id[entity]] = text
            else:
                pass
    entity2text = OrderedDict(sorted(entity2text.items(), key=lambda x: x[0]))

    relation2text = {}
    with open(os.path.join(args.data_path, "relation2text.txt"), 'r', encoding='utf-8') as fin:
        for l in fin:
            relation, text = l.strip().split('\t')
            if 'FB15k' in args.data_path:
                relation2text[rel2id['+{}'.format(relation)]] = text
                relation2text[rel2id['-{}'.format(relation)]] = "inverse " + text
            elif 'NELL' in args.data_path:
                relation2text[rel2id[relation]] = text
            # print()
    relation2text = OrderedDict(sorted(relation2text.items(), key=lambda x: x[0]))

    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    return entity2text_inputs, \
        entity2text, relation2text, \
        train_queries, train_answers, \
        valid_queries, valid_hard_answers, valid_easy_answers, \
        test_queries, test_hard_answers, test_easy_answers

def update_args(model, dataset, args):
    args.pattern = '1-step'
    if model == 'vec':
        args.data_path = f'data/{dataset}'
        args.negative_sample_size = 128
        args.hidden_dim = 800
        args.gamma = 24
        args.lr = 0.0001
        args.batch_size = 512
        args.cpu_num = 0
        args.geo = 'vec'
        args.tasks = '1p.2p.3p.2i.3i.ip.pi.2u.up'
        args.betam = '(1600,2)'
        args.boxm = "(none,0.02)"
        args.valid_steps = 15000
        args.max_steps = 450001
        # args.checkpoint_path = f'./saved_logs/{dataset}/1p.2p.3p.2i.3i.ip.pi.2u.up/{args.geo}_fol/g-24/2024_09_24'
    elif model == 'box':
        args.data_path = f'data/{dataset}'
        args.negative_sample_size = 128
        args.hidden_dim = 400
        args.gamma = 24
        args.lr = 0.0001
        args.batch_size = 512
        args.cpu_num = 0
        args.geo = 'box'
        args.tasks = '1p.2p.3p.2i.3i.ip.pi.2u.up'
        args.betam = '(1600,2)'
        args.boxm = "(none,0.02)"
        args.valid_steps = 15000
        args.max_steps = 450001
    elif model == 'beta':
        args.data_path = f'data/{dataset}'
        args.negative_sample_size = 128
        args.hidden_dim = 400
        args.gamma = 60
        args.lr = 0.0001
        args.batch_size = 512
        args.cpu_num = 0
        args.geo = 'beta'
        args.tasks = '1p.2p.3p.2i.3i' #'1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up' 'pni'
        args.betam = '(1600,2)'
        args.boxm = "(none,0.02)"
        args.valid_steps = 15000
        args.max_steps = 450001
        args.checkpoint_path = f'./saved_logs/{dataset}/1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up/{args.geo}/g-60-mode-(1600,2)/2024_01_18'
    elif model == 'cone':
        args.data_path = f'data/{dataset}'
        args.negative_sample_size = 128
        args.hidden_dim = 800
        args.gamma = 30
        args.learning_rate = 0.00005
        args.batch_size = 512
        args.cpu_num = 0
        args.geo = 'cone'
        args.tasks = '1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up' #'1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up' 'pni'
        args.cone_center_reg = 0.02
        args.cone_drop = 0.1
        args.valid_steps = 30000
        args.max_steps = 300001
    elif model == 'fuzzy':
        args.data_path = f'data/{dataset}'
        args.negative_sample_size = 128
        args.hidden_dim = 1000
        args.gamma = 0.5
        args.lr = 0.0005
        args.batch_size = 512
        args.cpu_num = 0
        args.geo = 'fuzzy'
        args.tasks = '1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up' #'1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up' 'pni'
        args.betam = '(1600,2)'
        args.boxm = "(none,0.02)"
        args.valid_steps = 5000
        args.max_steps = 450001
        args.fuzzy_gamma_coff = 20
        args.fuzzy_margin_type = 'logsigmoid_bpr'
        args.fuzzy_load_pretrained = False  # only Nell is True
        args.fuzzy_regularizer = '01'
        args.fuzzy_lr_scheduler = 'annealing'
        args.fuzzy_optimizer = 'AdamW'
        args.fuzzy_L2_reg = 5e-2
        args.fuzzy_projection_type = 'rtransform'
        args.fuzzy_num_rel_base = 30
    elif model == 'mlp2vec':
        args.data_path = f'data/{dataset}'
        args.negative_sample_size = 128
        args.hidden_dim = 800
        args.gamma = 24
        args.lr = 0.0001
        args.batch_size = 512
        args.cpu_num = 10
        args.geo = 'mlp2vec'
        args.tasks = '1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up' #'1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up' 'pni'
        args.valid_steps = 15000
        args.max_steps = 450001
        args.mlp_mode = 1
    else:
        raise ValueError('model must be in [vec, box, beta, cone, fuzzy, mlp2vec]')

    return args

def main(args):
    # dataset = 'FB15k-237-betae'   # NELL  FB15k-237
    # model_name = 'vec'
    # model_name = 'box'
    # model_name = 'beta'
    # model_name = 'cone'
    # model_name = 'fuzzy'
    # model_name = 'mlp2vec'

    # model_name = 'vec_temp'
    # model_name = 'box_temp'
    # model_name = 'beta_temp'

    dataset = args.dataset
    model_name = args.model_name

    args = update_args(model=model_name, dataset=dataset, args=args)
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta', "only BetaE supports modeling union using De Morgan's Laws"

    cur_time = parse_time().split('-')[0]
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix

    print("overwritting args.save_path")
    args.save_path = os.path.join(prefix, args.data_path.split('/')[-1], args.tasks, args.geo)
    if args.geo in ['box', 'box_temp']:
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.box_mode)
    elif args.geo in ['vec', 'mlp2vec', 'vec_temp']:
        tmp_str = "g-{}".format(args.gamma)
    elif args.geo in ['beta', 'beta_temp']:
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.beta_mode)
    elif args.geo == 'cone':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.cone_center_reg)
    elif args.geo == 'fuzzy':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.fuzzy_gamma_coff)

    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path
    else:
        cur_time = cur_time.replace('.', '_')
        cur_time = cur_time.replace(':', '_')
        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("logging to", args.save_path)
    if not args.do_train:  # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    set_logger(args)

    with open('%s/stats.txt' % args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
        ntype = int(entrel[2].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('-------------------------------' * 3)
    logging.info('Geo: %s' % args.geo)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)

    entity2text_inputs, \
        entity2text, relation2text, \
        train_queries, train_answers, \
        valid_queries, valid_hard_answers, valid_easy_answers, \
        test_queries, test_hard_answers, test_easy_answers = load_data(args, tasks)

    logging.info("Training info:")
    if args.do_train:
        for query_structure in train_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)
        path_list = ['1p', '2p', '3p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_queries_text = pickle.load(open(os.path.join(args.data_path, "train-path-queries-text.pkl"), 'rb'))
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(queries=train_path_queries,
                         query2text=train_path_queries_text,
                         entity2text_inputs=entity2text_inputs,
                         entity2text=entity2text,
                         relation2text=relation2text,
                         nentity=nentity,
                         nrelation=nrelation,
                         negative_sample_size=args.negative_sample_size,
                         answer=train_answers,
                         args=args),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_queries_text = pickle.load(
                open(os.path.join(args.data_path, "train-other-queries-text.pkl"), 'rb'))
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(queries=train_other_queries,
                             query2text=train_other_queries_text,
                             entity2text_inputs=entity2text_inputs,
                             entity2text=entity2text,
                             relation2text=relation2text,
                             nentity=nentity,
                             nrelation=nrelation,
                             negative_sample_size=args.negative_sample_size,
                             answer=train_answers,
                             args=args),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None

    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_queries_text = pickle.load(open(os.path.join(args.data_path, "valid-queries-text.pkl"), 'rb'))
        valid_dataloader = DataLoader(
            TestDataset(
                valid_queries,
                valid_queries_text,
                entity2text_inputs,
                entity2text,
                relation2text,
                args.nentity,
                args.nrelation,
                args
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    logging.info("Test info:")
    if args.do_test:
        # test_queries = {((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)):
        #                     test_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))]}
        for query_structure in test_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_queries_text = pickle.load(open(os.path.join(args.data_path, "test-queries-text.pkl"), 'rb'))
        test_dataloader = DataLoader(
            TestDataset(
                test_queries,
                test_queries_text,
                entity2text_inputs,
                entity2text,
                relation2text,
                args.nentity,
                args.nrelation,
                args
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    if args.geo in ['vec', 'box', 'beta', 'cone']:
        model = KGReasoning(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            geo=args.geo,
            use_cuda=args.cuda,
            box_mode=eval_tuple(args.box_mode),
            beta_mode=eval_tuple(args.beta_mode),
            test_batch_size=args.test_batch_size,
            query_name_dict=query_name_dict,
            args=args
        )
    elif args.geo == 'fuzzy':
        model = KGFuzzyReasoning(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            geo=args.geo,
            use_cuda=args.cuda,
            box_mode=eval_tuple(args.box_mode),
            beta_mode=eval_tuple(args.beta_mode),
            test_batch_size=args.test_batch_size,
            query_name_dict=query_name_dict,
            logic_type=args.fuzzy_logic,
            gamma_coff=args.fuzzy_gamma_coff,
            regularizer_setting={
                'type': args.fuzzy_regularizer,  # for query
                'e_reg_type': args.fuzzy_regularizer if args.fuzzy_e_regularizer == 'same' else args.fuzzy_e_regularizer,
                'prob_dim': args.fuzzy_prob_dim,  # for matrix softmax
                'dual': True if args.fuzzy_loss_type == 'weighted_fuzzy_containment' else False,
                'e_layernorm': args.fuzzy_entity_ln_before_reg  # apply Layer Norm before next step's regularizer
            },
            loss_type=args.fuzzy_loss_type,
            margin_type=args.fuzzy_margin_type,
            device=torch.device('cuda' if args.cuda else 'cpu'),
            godel_gumbel_beta=args.fuzzy_godel_gumbel_beta,
            gumbel_temperature=args.fuzzy_gumbel_temperature,
            projection_type=args.fuzzy_projection_type,
            args=args
        )
    elif args.geo == 'mlp2vec':
        model = MLP2Vec(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            geo=args.geo,
            use_cuda=args.cuda,
            mlp_mode=args.mlp_mode,
            test_batch_size=args.test_batch_size,
            query_name_dict=query_name_dict,
            args=args
        )

    if args.cuda:
        model = model.cuda()

    if not args.fine_tuning:
        for name, param in model.know_injector.ke.plm.named_parameters():
            if 'memory' in name:
                print('trained:', name)
            else:
                print('frozen:', name)
                param.requires_grad = args.fine_tuning
    print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.do_train:
        if args.geo in ['vec', 'box', 'beta', 'cone', 'mlp2vec', 'vec_temp', 'box_temp', 'beta_temp']:
            current_learning_rate = args.learning_rate
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=current_learning_rate
            )
            warm_up_steps = args.max_steps // 2
            if args.geo == 'mlp2vec':
                torch.autograd.set_detect_anomaly(True)
        elif args.geo == 'fuzzy':
            current_learning_rate = args.learning_rate
            warm_up_steps = 0
            if args.fuzzy_optimizer == 'AdamW':  # use together with lr_scheduler none
                if args.fuzzy_L2_reg > 0:
                    weight_decay = args.fuzzy_L2_reg
                else:
                    weight_decay = 1e-2
                print(f'AdamW weight decay: {weight_decay}')
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, list(model.parameters())),
                    lr=args.learning_rate,
                    eps=1e-06,
                    weight_decay=weight_decay
                )
            else:
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, list(model.parameters())),
                    lr=current_learning_rate,
                    weight_decay=args.fuzzy_L2_reg  # L2 regularization
                )

            if args.fuzzy_lr_scheduler == 'original':
                warm_up_steps = args.max_steps // 2  # reduce lr when reaching warm up steps
            elif args.fuzzy_lr_scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.2)
            elif args.fuzzy_lr_scheduler == 'annealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=0,
                                                                       last_epoch=-1)
            elif args.fuzzy_lr_scheduler == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=args.valid_steps * 2,
                    verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
                    min_lr=0.0001, eps=1e-07
                )
            elif args.fuzzy_lr_scheduler == 'onecycle':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear',
                                                                final_div_factor=10,
                                                                max_lr=5e-4,
                                                                total_steps=args.batch_size * args.max_steps + 1)

    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0

    step = init_step
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %d' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    if args.do_train:
        training_path_logs = []
        training_other_logs = []

        # test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict,
        #                             'Test', step, writer)

        # #Training Loop
        for step in range(init_step, args.max_steps):
            if step == 2 * args.max_steps // 3:
                args.valid_steps *= 4

            log_path = model.train_step(model, optimizer, train_path_iterator, args, step)
            for metric in log_path:
                writer.add_scalar('path_' + metric, log_path[metric], step)
            if train_other_iterator is not None:
                log_other = model.train_step(model, optimizer, train_other_iterator, args, step)
                for metric in log_other:
                    writer.add_scalar('other_' + metric, log_other[metric], step)
                log_path = model.train_step(model, optimizer, train_path_iterator, args, step)

                training_other_logs.append(log_other)

            training_path_logs.append(log_path)

            if args.geo in ['vec', 'box', 'beta', 'cone', 'mlp2vec']:
                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 5
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * 1.5
            elif args.geo == 'fuzzy':
                if args.fuzzy_lr_scheduler != 'none':  # do not change lr if 'none'
                    if args.fuzzy_lr_scheduler == 'original':  # BetaE original
                        if step >= warm_up_steps:
                            current_learning_rate = current_learning_rate / 5
                            warm_up_steps = warm_up_steps * 1.5
                            optimizer = torch.optim.Adam(
                                filter(lambda p: p.requires_grad, model.parameters()),
                                lr=current_learning_rate
                            )  # new optimizer

                    elif args.fuzzy_lr_scheduler in ('step', 'annealing', 'plateau', 'onecycle'):
                        if args.fuzzy_lr_scheduler == 'plateau':
                            scheduler.step(log_other['loss'])
                        else:
                            scheduler.step()

            # if step % args.save_checkpoint_steps == 0:
            #     save_variable_list = {
            #         'step': step,
            #         'current_learning_rate': current_learning_rate,
            #         'warm_up_steps': warm_up_steps
            #     }
            #     save_model(model, optimizer, save_variable_list, args)

            if step % args.valid_steps == 0 and step > 0:
                # if args.do_valid:
                #     logging.info('Evaluating on Valid Dataset...')
                #     valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args, valid_dataloader,
                #                                  query_name_dict, 'Valid', step, writer)

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader,
                                                query_name_dict, 'Test', step, writer)
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_path_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_path_logs]) / len(training_path_logs)
                log_metrics('Training average path', step, metrics)

                metrics = {}
                for metric in training_other_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_other_logs]) / len(training_other_logs)
                log_metrics('Training average other', step, metrics)

                logging.info('step: {}, current_learning_rate: {}, warm_up_steps: {}'.format(step,
                                                                                             current_learning_rate,
                                                                                             warm_up_steps))

                logging.info('=================================================================')

                training_path_logs = []
                training_other_logs = []

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args)

    try:
        print(step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader,
                                    query_name_dict, 'Test', step, writer)

    logging.info("Training finished!!")


if __name__ == '__main__':
    # python main.py --dataset NELL-betae --model_name mlp2vec
    main(parse_args())