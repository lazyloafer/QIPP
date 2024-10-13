# QIPP + [CQD-Beam](https://openreview.net/forum?id=Mos9F9kDwkz)

## 1. Datasets

Download `FB15k-237-betae`, `FB15k-betae`, and `NELL-betae` from [BetaE](https://arxiv.org/abs/2010.11465).

Put these three datasets into ``./data/``:
```
./data/
│
├── FB15k-237-betae/
│   ├── train + valid + test data
├── FB15k-betae/
│   ├── train + valid + test data
└── NELL-betae/
    └── train + valid + test data
```

## 2. Pre-trained Language Model

Download the Pre-trained BERT from [Hugging Face](https://huggingface.co/google-bert/bert-base-cased/tree/main)

Put Pre-trained BERT files into ``./PLM/bert-base-cased/``

Set `"num_hidden_layers"` in ``./PLM/bert-base-cased/config.json`` to `1`


## 3. Training
```
python main.py --dataset FB15k-237-betae --tasks 1p
python main.py --dataset FB15k-betae --tasks 1p
python main.py --dataset NELL-betae --tasks 1p
```

## 4. Answer the complex queries
### 4.1. FB15k-237
```
python main.py --dataset FB15k-237-betae --tasks 1p --checkpoint_path ./logs/FB15k-237-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-237-betae --tasks 2p --checkpoint_path ./logs/FB15k-237-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-237-betae --tasks 3p --checkpoint_path ./logs/FB15k-237-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-237-betae --tasks 2i --checkpoint_path ./logs/FB15k-237-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-237-betae --tasks 3i --checkpoint_path ./logs/FB15k-237-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-237-betae --tasks ip --checkpoint_path ./logs/FB15k-237-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-237-betae --tasks pi --checkpoint_path ./logs/FB15k-237-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-237-betae --tasks 2u --checkpoint_path ./logs/FB15k-237-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-237-betae --tasks up --checkpoint_path ./logs/FB15k-237-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
```

### 4.2. NELL995
```
python main.py --dataset NELL-betae --tasks 1p --checkpoint_path ./logs/NELL-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset NELL-betae --tasks 2p --checkpoint_path ./logs/NELL-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset NELL-betae --tasks 3p --checkpoint_path ./logs/NELL-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset NELL-betae --tasks 2i --checkpoint_path ./logs/NELL-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset NELL-betae --tasks 3i --checkpoint_path ./logs/NELL-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset NELL-betae --tasks ip --checkpoint_path ./logs/NELL-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset NELL-betae --tasks pi --checkpoint_path ./logs/NELL-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset NELL-betae --tasks 2u --checkpoint_path ./logs/NELL-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset NELL-betae --tasks up --checkpoint_path ./logs/NELL-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
```

### 4.3. FB15k
```
python main.py --dataset FB15k-betae --tasks 1p --checkpoint_path ./logs/FB15k-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-betae --tasks 2p --checkpoint_path ./logs/FB15k-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-betae --tasks 3p --checkpoint_path ./logs/FB15k-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-betae --tasks 2i --checkpoint_path ./logs/FB15k-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-betae --tasks 3i --checkpoint_path ./logs/FB15k-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-betae --tasks ip --checkpoint_path ./logs/FB15k-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-betae --tasks pi --checkpoint_path ./logs/FB15k-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-betae --tasks 2u --checkpoint_path ./logs/FB15k-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
python main.py --dataset FB15k-betae --tasks up --checkpoint_path ./logs/FB15k-betae/1p/cqd/g-cqd/{YYYY.MM.DD}
```
