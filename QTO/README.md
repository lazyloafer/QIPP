## QTO + QIPP

### 1. Datasets

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

### 2. Pre-trained Language Model

Download the Pre-trained BERT from [Hugging Face](https://huggingface.co/google-bert/bert-base-cased/tree/main)

Put Pre-trained BERT files into ``kbc/src/PLM/bert-base-cased/``

Set `"num_hidden_layers"` in ``kbc/src/PLM/bert-base-cased/config.json`` to `1`

### 3. Pretrain ComplEx + QIPP
```
cd kbc\src
python main.py --dataset FB15K-237 --score_rel False --score_lhs True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 100
python main.py --dataset NELL995 --score_rel False --score_lhs True --model ComplEx --rank 1000 --learning_rate 0.05 --batch_size 1000 --lmbda 0.05 -w_rel 0 --max_epochs 100
python main.py --dataset FB15K --score_rel False --score_lhs True --model ComplEx --rank 1000 --learning_rate 0.05 --batch_size 1000 --lmbda 0.01 --w_rel 0.1 --max_epochs 100
```

### 4. Test on QTO
```
cd ..
cd ..
python main.py --dataset FB15k-237
python main.py --dataset NELL995
python main.py --dataset FB15K
```
