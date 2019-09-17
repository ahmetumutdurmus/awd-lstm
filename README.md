# Regularizing and Optimizing LSTM Language Models by Merity et al. (2017).
This repository contains the replication of "Regularizing and Optimizing LSTM Language Models" by Merity et al. (2017).

The AWD-LSTM model introduced in the paper still forms the basis for the state-of-the-art results in language modeling on smaller benchmark datasets such as the Penn Treebank and WikiText-2 according to the [NLP-Progress](https://nlpprogress.com/english/language_modeling.html) repository. On bigger datasets, such as WikiText-103 and Google One Billion Word Benchmark, the state-of-the-art is generally achieved with introducing some form of attention to the model. Generally this is some variant of the [Transformer](https://arxiv.org/abs/1706.03762) model. This could likely be explained by the fact that attention models tend to have greater number of parameters and can overfit the data more easily. 

The original paper can be found at: [https://arxiv.org/abs/1708.02182](https://arxiv.org/abs/1708.02182)  
While the original code written in Python 3 and PyTorch 0.4 can be found at: [https://github.com/salesforce/awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm)

I have replicated the paper using Python 3.7 and PyTorch 1.2 with CUDA Toolkit 10.0. So the model is now PyTorch 1.2 compatible.

The repository contains four scripts:

+ `model.py` contains the model described as in the paper.
+ `ntasgd.py` contains the NT-ASGD optimizer described as in the paper.
+ `main.py` is used to replicate the main results in the paper. 
+ `finetune.py` is used to replicate the finetuning process in the paper. 

## Experiments
The experiments run on the two different word-level datasets can be replicated from the terminal as follows: 

### Word Level Penn Treebank (PTB)
+ `python main.py --data PTB --save model.tar --layer_num 3 --embed_size 400 --hidden_size 1150 --lstm_type pytorch --w_drop 0.5 --dropout_i 0.4 --dropout_l 0.3 --dropout_o 0.4 --dropout_e 0.1 --winit 0.1 --batch_size 40 --bptt 70 --ar 2 --tar 1 --weight_decay 1.2e-6 --epochs 750 --lr 30 --max_grad_norm 0.25 --non_mono 5 --device gpu --log 100`
+ `python finetune.py --data PTB --load model.tar --layer_num 3 --embed_size 400 --hidden_size 1150 --lstm_type pytorch --w_drop 0.5 --dropout_i 0.4 --dropout_l 0.3 --dropout_o 0.4 --dropout_e 0.1 --winit 0.1 --batch_size 40 --bptt 70 --ar 2 --tar 1 --weight_decay 1.2e-6 --lr 30 --max_grad_norm 0.25 --non_mono 5 --device gpu --log 100`

### Word Level WikiText-2 (WT2)
+ `python main.py --data PTB --save model.tar --layer_num 3 --embed_size 400 --hidden_size 1150 --lstm_type pytorch --w_drop 0.65 --dropout_i 0.4 --dropout_l 0.3 --dropout_o 0.4 --dropout_e 0.1 --winit 0.1 --batch_size 80 --bptt 70 --ar 2 --tar 1 --weight_decay 1.2e-6 --epochs 750 --lr 30 --max_grad_norm 0.25 --non_mono 5 --device gpu --log 50`
+ `python finetune.py --data PTB --load model.tar --layer_num 3 --embed_size 400 --hidden_size 1150 --lstm_type pytorch --w_drop 0.65 --dropout_i 0.4 --dropout_l 0.3 --dropout_o 0.4 --dropout_e 0.1 --winit 0.1 --batch_size 80 --bptt 70 --ar 2 --tar 1 --weight_decay 1.2e-6 --lr 30 --max_grad_norm 0.25 --non_mono 5 --device gpu --log 50`

Couple of things to note:

You can use both my implementation of LSTM by setting `--lstm_type custom` or the PyTorch's embedded C++ implementation using `--lstm_type pytorch`. PyTorch's implementation is about 2 times faster.

You can interrupt the training or finetuning process at any time without losing your model with your keyboard using `Ctrl-C`. I have implemented the relevant error catching code in the fashion of the original authors.

Finally note that `finetune.py` overwrites the model it loads. If you wish to keep the original model, copy it elsewhere before starting the finetuning. 
