# Regularizing and Optimizing LSTM Language Models by Merity et al. (2017).

This repository contains the replication of 'Regularizing and Optimizing LSTM Language Models' by Merity et al. (2017).

The paper can be found at: [https://arxiv.org/abs/1708.02182](https://arxiv.org/abs/1708.02182)  
While the original code written in Python 3 and PyTorch 0.4 can be found at: [https://github.com/salesforce/awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm)

I have replicated the paper using Python 3.7 and PyTorch 1.2 with CUDA Toolkit 10.0. So the model is now PyTorch 1.2 compatible.

The repository contains four scripts:

+ `model.py` contains the model described as in the paper.
+ `ntasgd.py` contains the NT-ASGD optimizer described as in the paper.
+ `main.py` is used to replicate the main results in the paper. 
+ `finetune.py` is used to replicate the finetuning process in the paper. 

I have not implemented the [continious cache pointer](https://arxiv.org/abs/1612.04426).


## Experiments

### Word Level Penn Treebank (PTB)
+ `python main.py --data PTB --save model.tar --layer_num 3 --embed_size 400 --hidden_size 1150 --lstm_type pytorch --w_drop 0.5 --dropout_i 0.4 --dropout_l 0.3 --dropout_o 0.4 --dropout_e 0.1 --winit 0.1 --batch_size 40 --bptt 70 --ar 2 --tar 1 --weight_decay 1.2e-6 --epochs 750 --lr 30 --max_grad_norm 0.25 --non_mono 5 --device gpu`
+ `python ensemble.py --data PTB --load model.tar --layer_num 3 --embed_size 400 --hidden_size 1150 --lstm_type pytorch --w_drop 0.5 --dropout_i 0.4 --dropout_l 0.3 --dropout_o 0.4 --dropout_e 0.1 --winit 0.1 --batch_size 40 --bptt 70 --ar 2 --tar 1 --weight_decay 1.2e-6 --lr 30 --max_grad_norm 0.25 --non_mono 5 --device gpu`
