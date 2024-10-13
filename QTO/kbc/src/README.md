## Pretrain ComplEx with QIPP

```
python main.py --dataset FB15K-237 --score_rel False --score_lhs True --model ComplEx --rank 1000 --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 100
python main.py --dataset NELL995 --score_rel False --score_lhs True --model ComplEx --rank 1000 --learning_rate 0.05 --batch_size 1000 --lmbda 0.05 -w_rel 0 --max_epochs 100
python main.py --dataset FB15K --score_rel False --score_lhs True --model ComplEx --rank 1000 --learning_rate 0.05 --batch_size 1000 --lmbda 0.01 --w_rel 0.1 --max_epochs 100
```