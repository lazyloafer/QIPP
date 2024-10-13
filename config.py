import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)

    parser.add_argument('--cuda', default=True, help='use GPU')

    parser.add_argument('--do_train', default=True, help="do train")
    parser.add_argument('--do_valid', default=False, help="do valid")
    parser.add_argument('--do_test', default=True, help="do test")

    parser.add_argument('--data_path', type=str, default='data/FB15k-237-betae', help="KG data path")  # FB15k-237  NELL
    parser.add_argument('-n', '--negative_sample_size', default=2, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=400, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=24, type=float, help="margin in the loss") # betae 60, box vec 24
    parser.add_argument('-b', '--batch_size', default=512, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=0, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str,
                        help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=450001, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int,
                        help="no need to set manually, will configure automatically")

    parser.add_argument('--save_checkpoint_steps', default=50000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=15000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--geo', default='box', type=str, choices=['vec', 'box', 'beta', 'cone', 'fuzzy'],
                        help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')
    parser.add_argument('--print_on_screen', default=True, type=bool, help='print log on screen')

    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str,
                        help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    # # 1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str,
                        help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str,
                        help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')

    # # special for ConE
    parser.add_argument('-cone_cenr', '--cone_center_reg', default=0.02, type=float,
                        help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
    parser.add_argument('--cone_drop', type=float, default=0.2, help='dropout rate')

    # # special for FuzzQE
    parser.add_argument('--fuzzy_margin_type', default='logsigmoid_bpr', type=str,
        choices=[
            'logsigmoid', 'logsigmoid_bpr', 'logsigmoid_bpr_digits', 'bpr_digits', 'logsigmoid_avg', 'bpr', 'softmax', 'nll'
        ],
        help='ways to implement margin')
    parser.add_argument('--fuzzy_projection_type', default='rtransform', type=str, choices=['mlp', 'rtransform', 'transe'])
    parser.add_argument('--fuzzy_num_rel_base', default=30, type=int)
    parser.add_argument('--fuzzy_regularizer', default='01', type=str,
                        choices=['01', 'vector_softmax', 'matrix_softmax', 'matrix_L1', 'matrix_sigmoid_L1', 'sigmoid',
                                 'vector_sigmoid_L1'],
                        help='ways to regularize parameters')
    parser.add_argument('--fuzzy_logic', default='godel', type=str, choices=['luka', 'godel', 'product', 'godel_gumbel'],
                        help='fuzzy logic type')
    parser.add_argument('-k', '--fuzzy_prob_dim', default=8, type=int,
                        help="for matrix_softmax and matrix_L1. dims per prob vector")
    parser.add_argument('--fuzzy_e_regularizer', default='same', type=str,
                        choices=['same', '01', 'vector_softmax', 'matrix_softmax', 'matrix_L1', 'matrix_sigmoid_L1',
                                 'sigmoid', 'vector_sigmoid_L1'],
                        help='set regularizer for entities, different from queries')  # if 'same' (default), just use args.regularizer
    parser.add_argument('--fuzzy_gamma_coff', default=20, type=float, help='coefficient for gamma')
    parser.add_argument('--fuzzy_lr_scheduler', default='annealing', type=str,
                        choices=['none', 'original', 'step', 'annealing', 'plateau', 'onecycle'])
    parser.add_argument('--fuzzy_optimizer', default='AdamW', type=str, choices=['Adam', 'AdamW'])
    parser.add_argument('--fuzzy_L2_reg', default=5e-2, type=float)
    parser.add_argument('--fuzzy_load_pretrained', default=False,
                        help='load pretrained embeddings. dimension=1000. only for NELL')
    parser.add_argument('--fuzzy_loss_type', default='cos', type=str,
                        choices=['cos',
                                 'cos_digits', 'L1_cos_digits', 'dot_layernorm_digits',
                                 'dot', 'weighted_dot',
                                 'soft_min_digits',
                                 'kl', 'entropy',
                                 'discrete_cos', 'discrete_prob', 'discrete_gumbel', 'gumbel_softmax',
                                 'fuzzy_containment', 'weighted_fuzzy_containment',
                                 'entity_multinomial_dot',
                                 # use with sigmoid regularizer. L1 noramlize entity before computing score
                                 'normalized_entity_dot'
                                 # normalize entity when no grad. use with 0/1 regularizer for entity, and sigmoid for query
                                 ], help="loss type")
    parser.add_argument('--fuzzy_entity_ln_before_reg', action="store_true",
                        help='apply layer normalization before applying regularizer to entities')
    parser.add_argument('--fuzzy_godel_gumbel_beta', default=0.01, type=int,
                        help="Gumbel beta for min/max computation when logic=godel_gumbel")
    parser.add_argument('--fuzzy_gumbel_temperature', default=1, type=float,
                        help="Gumbel temperature for gumbel softmax")
    parser.add_argument('--fuzzy_no_anchor_reg', action='store_true', help='no anchor entity regularizer')
    parser.add_argument('--fuzzy_simplE', action="store_true", help="Use different head and tail embeddings for entities")
    parser.add_argument('--fuzzy_use_attention', action='store_true', help='use attention for conjunction')
    parser.add_argument('--fuzzy_gumbel_attention', default='none', type=str, choices=['none', 'plain', 'query_dependent'],
                        help="Add distribution-wise attention")
    parser.add_argument('--fuzzy_query_unnorm', action="store_true")
    parser.add_argument('--fuzzy_in_batch_negative', action='store_true', help='use in-batch negatives')
    parser.add_argument(
        '--fuzzy_with_counter', action="store_true", help="add neg q into negative samples"
    )

    # # special for MLP2Vec
    parser.add_argument(
        "-nlayers",
        "--mlp_mode",
        default=1,
        type=int,
        help="Number of layers (before last) for the neural networs",
    )

    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'],
                        help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')


    # # special for TEMP
    parser.add_argument('--neighbor_ent_type_samples', default=32, type=int, help='number of sampled entity type')
    parser.add_argument('--neighbor_rel_type_samples', default=64, type=int, help='number of sampled relation type')
    parser.add_argument('-cenr', '--center_reg', default=0.02, type=float)

    # # special for our PLM plugin
    parser.add_argument('--plm_name', default='./PLM/bert-base-cased', type=str, help='path for loading PLM model')
    # parser.add_argument('--plm_num_layers', default=6, type=int, help='use layers in plm')
    parser.add_argument('--max_seq_len', default=512, type=int, help='max_seq_len')
    parser.add_argument('--fine_tuning', default=True, type=bool, help='fine tuning plm')
    parser.add_argument('--use_memory', default=False, type=bool, help='whether use memory/prefix in PLM')
    parser.add_argument('--memory_size', default=20, type=int, help='memory/prefix size in PLM')

    parser.add_argument('--extractor_size', default=20, type=int, help='extractor size in KnowExtractor')
    parser.add_argument('--extractor_num_attention_heads', default=8, type=int, help='attention head in extractor')
    parser.add_argument('--extractor_attention_probs_dropout_prob', default=0.1, type=int,
                        help='attention dropout in extractor')

    parser.add_argument('--KnowEncoder_layer_num', default=1, type=int, help='KnowEncoder_layer_num')


    return parser.parse_args(args)