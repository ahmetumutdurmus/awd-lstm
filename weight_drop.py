import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit

#import os
#os.chdir('C:\\Users\\adurmus\\Desktop\\awd-lstm')
import model as m


class ModelConfig(object):
    vocab_size = 10000
    embed_size = 400
    hidden_size = 1150
    dropout_embed = 0.1
    weight_drop = 0.5
    dropout_i = 0.4
    dropout_l = 0.25
    dropout_y = 0.4
    init_e = 0.1
    epochs = 500
    bptt = 70
    batch_size = 20
    eval_batch_size = 10
    test_batch_size = 1
    lr = 30
    ar = 2
    tar = 1
    max_norm = 0.25
    weight_decay = 1.2e-6

def save_model(model, optimizer, epoch = None):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, 'model.tar')

def data_init():
    with open("./data/ptb.train.txt") as f:
        file = f.read()
        trn = file[1:].split(' ')
    with open("./data/ptb.valid.txt") as f:
        file = f.read()
        vld = file[1:].split(' ')
    with open("./data/ptb.test.txt") as f:
        file = f.read()
        tst = file[1:].split(' ')
    words = sorted(set(trn))
    ind2char = {i: c for i, c in enumerate(words)}
    char2ind = {c: i for i, c in enumerate(words)}
    trn = [char2ind[c] for c in trn]
    vld = [char2ind[c] for c in vld]
    tst = [char2ind[c] for c in tst]
    return torch.tensor(trn,dtype=torch.int64).reshape(-1, 1), torch.tensor(vld,dtype=torch.int64).reshape(-1, 1), torch.tensor(tst,dtype=torch.int64).reshape(-1, 1), ind2char

def get_seq_len(bptt = 70):
        seq_len = bptt if np.random.random() < 0.95 else bptt/2
        seq_len = round(np.random.normal(seq_len, 5))
        while seq_len <= 5 or seq_len >= 90:
            seq_len = bptt if np.random.random() < 0.95 else bptt/2
            seq_len = round(np.random.normal(seq_len, 5))
        return seq_len

def batchify(data, batch_size):
    num_batches = data.size(0)//batch_size
    data = data[:num_batches*batch_size]
    return data.reshape(batch_size, -1).transpose(1, 0)

def minibatch(data, seq_length):
    num_batches = data.size(0)
    dataset = []
    for i in range(0, num_batches-1, seq_length):
        ls = min(i+seq_length, num_batches-1)
        x = data[i:ls,:]
        y = data[i+1:ls+1,:]
        dataset.append((x, y))
    return dataset


class NTASGD(optim.Optimizer):
    def __init__(self, params, lr=1, n=5, weight_decay=0, fine_tuning=False):
        t0 = 0 if fine_tuning else 10e7
        defaults = dict(lr=lr, n=n, weight_decay=weight_decay, fine_tuning=fine_tuning, t0=t0, t=0, logs=[])
        super(NTASGD, self).__init__(params, defaults)

    def check(self, v):
        for group in self.param_groups:
            #Training
            if (not group['fine_tuning'] and group['t0'] == 10e7) or (group['fine_tuning']):
                if group['t'] > group['n'] and v > min(group['logs'][:-group['n']]):
                    group['t0'] = self.state[next(iter(group['params']))]['step']
                    print("Non-monotonic condition is triggered!")
                    return True
                group['logs'].append(v)
                group['t'] += 1

    def lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
                               
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
#                if p.grad is None:
 #                   continue
                grad = p.grad.data
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(p.data)    
                state['step'] += 1
                # update parameter
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                p.data.add_(-group['lr'], grad)
                # averaging
                if state['mu'] != 1:
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)
                # update mu
                state['mu'] = 1 / max(1, state['step'] - group['t0'])


def perplexity(data, model):
    model.eval()
    data = minibatch(data, config.bptt)
    with torch.no_grad():
        losses = []
        batch_size = data[0][0].size(1)
        states = model.state_init(batch_size)
        for x, y in data:
            x = x.to("cuda:0")
            y = y.to("cuda:0")
            scores, states = model(x, states)
            loss = F.cross_entropy(scores, y.reshape(-1))
            #Again with the sum/average implementation described in 'nll_loss'.
            losses.append(loss.data.item())
    return np.exp(np.mean(losses))
    
def train(data, model, optimizer):
    trn, vld, tst = data
    tic = timeit.default_timer()
    total_words = 0
    print("Starting training.")
    best_val = config.vocab_size
#    epoch = 0
    for epoch in range(config.epochs):
        seq_len = get_seq_len(config.bptt)
        num_batch = ((trn.size(0)-1)// seq_len + 1)
        optimizer.lr(seq_len/config.bptt*config.lr)
        states = model.state_init(config.batch_size)
        model.train()
        for i, (x, y) in enumerate(minibatch(trn, seq_len)):
            x = x.to("cuda:0")
            y = y.to("cuda:0")
            total_words += x.numel()
            states = model.detach(states)
            scores, states, activations = model(x, states)
            loss = F.cross_entropy(scores, y.reshape(-1))
            h, h_m = activations
            ar_reg = config.ar * h_m.pow(2).mean()
            tar_reg = config.tar * (h[:-1] - h[1:]).pow(2).mean()
            loss_reg = loss + ar_reg + tar_reg
            loss_reg.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
            optimizer.step()
            optimizer.zero_grad()
            if i % (200) == 0:#num_batch//2:
                toc = timeit.default_timer()
                print("batch no = {:d} / {:d}, ".format(i, num_batch) +
                      "train loss = {:.3f}, ".format(loss.item()) +
                      "ar val = {:.3f}, ".format(ar_reg.item()) + 
                      "tar val = {:.3f}, ".format(tar_reg.item()) + 
                      "wps = {:d}, ".format(round(total_words/(toc-tic))) +
                      "dw.norm() = {:.3f}, ".format(norm) +
                      "lr = {:.3f}, ".format(seq_len/config.bptt*config.lr) + 
                      "since beginning = {:d} mins, ".format(round((toc-tic)/60)) +
                      "cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated()/1024/1024/1024))

        tmp = {}
        for (prm,st) in optimizer.state.items():
            tmp[prm] = prm.clone().detach()
            prm.data = st['ax'].clone().detach()
 #       for param in model.parameters():
 #           if 'ax' in optimizer.state[param]:
 #               with torch.no_grad():
 #               tmp[param] = param.clone().detach()
 #               param.data = optimizer.state[param]['ax'].clone().detach()

        val_perp = perplexity(vld, model)
        optimizer.check(val_perp)

        if val_perp < best_val:
            best_val = val_perp
            print("Best validation perplexity : {:.3f}".format(best_val))
            save_model(model, optimizer, epoch)
            print("Model saved!")

        for (prm, st) in optimizer.state.items():
            prm.data = tmp[prm].clone().detach()                
#        for param in model.parameters():
 #           if 'ax' in optimizer.state[param]:
 #               with torch.no_grad():
  #              param.data = tmp[param].clone().detach()                   

        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, val_perp))
        print("*************************************************\n")
        
    tmp = {}
    for (prm,st) in optimizer.state.items():
        tmp[prm] = prm.clone().detach()
        prm.data = st['ax'].clone().detach()
    
    tst_perp = perplexity(tst, model)
    print("Test set perplexity : {:.3f}".format(tst_perp))



torch.manual_seed(1110)
np.random.seed(1110)

config = ModelConfig()

trn, vld, tst, ind2char = data_init()

trn = batchify(trn, config.batch_size)
vld = batchify(vld, config.eval_batch_size)
tst = batchify(tst, config.test_batch_size)
#%%
model = m.Model(config)

model.to("cuda:0")
optimizer = NTASGD(model.parameters(), lr = config.lr, weight_decay = config.weight_decay)

train((trn, vld, tst), model, optimizer)

#checkpoint = torch.load('model.tar')
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
