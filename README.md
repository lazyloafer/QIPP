# QIPP
##1. QIPP + Iterative KGQE Models
**KGQE Models**
- [x] [CQD-Beam](https://arxiv.org/abs/2011.03459)
- [x] [QTO](https://proceedings.mlr.press/v202/bai23b.html)

Unzip ``CQD.zip`` and ``QTO.zip`` to the root. See ``CQD\README.md`` and ``QTO\README.md`` for details.

##2. QIPP + End2End KGQE Model
**KGQE Models**
- [x] [GQE](https://arxiv.org/abs/1806.01445)
- [x] [Q2B](https://arxiv.org/abs/2002.05969)
- [x] [BetaE](https://arxiv.org/abs/2010.11465)
- [x] [ConE](https://proceedings.neurips.cc/paper/2021/hash/a0160709701140704575d499c997b6ca-Abstract.html)
- [x] [MLP2Vec](https://arxiv.org/pdf/2209.14464)
- [x] [FuzzQE](https://ojs.aaai.org/index.php/AAAI/article/view/20310)

###2.1. Datasets

Download `FB15k-237-betae`, `FB15k-betae`, and `NELL-betae` from [BetaE](https://arxiv.org/abs/2010.11465).

Put these three datasets into ``data/``:
```
data/
│
├── FB15k-237-betae/
│   ├── train + valid + test data
├── FB15k-betae/
│   ├── train + valid + test data
└── NELL-betae/
    └── train + valid + test data
```

###2.2. Pre-trained Language Model

Download the Pre-trained BERT from [Hugging Face](https://huggingface.co/google-bert/bert-base-cased/tree/main)

Put Pre-trained BERT files into ``PLM/bert-base-cased/``

Set `"num_hidden_layers"` in ``PLM/bert-base-cased/config.json`` to `1`

###2.3. Training and Testing

Train QIPP with GQE:

```
python main.py --dataset NELL-betae --model_name geo
python main.py --dataset FB15k-237-betae --model_name geo
python main.py --dataset FB15k-betae --model_name geo
```

Train QIPP with Q2B:

```
python main.py --dataset NELL-betae --model_name box
python main.py --dataset FB15k-237-betae --model_name box
python main.py --dataset FB15k-betae --model_name box
```

Train QIPP with BetaE:

```
python main.py --dataset NELL-betae --model_name beta
python main.py --dataset FB15k-237-betae --model_name beta
python main.py --dataset FB15k-betae --model_name beta
```

Train QIPP with ConE:

```
python main.py --dataset NELL-betae --model_name cone
python main.py --dataset FB15k-237-betae --model_name cone
python main.py --dataset FB15k-betae --model_name cone
```

Train QIPP with MLP2Vec:

```
python main.py --dataset NELL-betae --model_name mlp2vec
python main.py --dataset FB15k-237-betae --model_name mlp2vec
python main.py --dataset FB15k-betae --model_name mlp2vec
```

Train QIPP with FuzzQE:

```
python main.py --dataset NELL-betae --model_name fuzzy
python main.py --dataset FB15k-237-betae --model_name fuzzy
python main.py --dataset FB15k-betae --model_name fuzzy
```
