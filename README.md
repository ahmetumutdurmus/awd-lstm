# Regularizing and Optimizing LSTM Language Models by Merity et al. (2017).

This repository is initialized to host a replication of 'Regularizing and Optimizing LSTM Language Models' by Merity et al. (2017). It will have a better readme once it is up and running. 

## Experiments

### Word Level Penn Treebank (PTB)
+ python main.py --layer_num 3 --embed_size 400 --hidden_size 1150 --lstm_type pytorch --w_drop 0.5 --dropout_i 0.4 --dropout_l 0.3 --dropout_o 0.4 --dropout_e 0.1 --winit 0.1 --batch_size 40 --bptt 70 --ar 2 --tar 1 --weight_decay 1.2e-6 --epochs 750 --lr 30 --max_grad_norm 0.25 --non_mono 5 --device gpu
