import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import timeit

from model import Model
from ntasgd import NTASGD

parser = argparse.ArgumentParser(description="Replication of Merity et al. (2017). \n https://arxiv.org/abs/1708.02182")
parser.add_argument("--load", type=str, default="model.tar", help="Where to find the model to finetune.")
parser.add_argument("--data", type=str, choices=["PTB","WT2"], default="PTB", help="The dataset to run the experiment on.")
parser.add_argument("--layer_num", type=int, default=3, help="The number of LSTM layers the model has.")
parser.add_argument("--embed_size", type=int, default=400, help="The number of hidden units per layer.")
parser.add_argument("--hidden_size", type=int, default=1150, help="The number of hidden units per layer.")
parser.add_argument("--lstm_type", type=str, choices=["pytorch","custom"], default="pytorch", help="Which implementation of LSTM to use."
                    + "Note that 'pytorch' is about 2 times faster.")
parser.add_argument("--w_drop", type=float, default=0.5, help="The weight drop parameter.")
parser.add_argument("--dropout_i", type=float, default=0.4, help="The dropout parameter on word vectors.")
parser.add_argument("--dropout_l", type=float, default=0.3, help="The dropout parameter between LSTM layers.")
parser.add_argument("--dropout_o", type=float, default=0.4, help="The dropout parameter on the last LSTM layer.")
parser.add_argument("--dropout_e", type=float, default=0.1, help="The dropout parameter on the embedding layer.")
parser.add_argument("--winit", type=float, default=0.1, help="The weight initialization parameter on the embedding layer.")
parser.add_argument("--batch_size", type=int, default=20, help="The batch size.")
parser.add_argument("--valid_batch_size", type=int, default=10, help="The batch size used on validation set.")
parser.add_argument("--test_batch_size", type=int, default=1, help="The batch size used on test set.")
parser.add_argument("--bptt", type=int, default=70, help="The sequence length parameter for bptt.")
parser.add_argument("--ar", type=float, default=2, help="The AR parameter.")
parser.add_argument("--tar", type=float, default=1, help="The TAR parameter.")
parser.add_argument("--weight_decay", type=float, default=1.2e-6, help="The weight decay parameter.")
parser.add_argument("--lr", type=float, default=30, help="The learning rate.")
parser.add_argument("--max_grad_norm", type=float, default=0.25, help="The maximum norm of gradients we impose on training.")
parser.add_argument("--non_mono", type=int, default=5, help="The masking length for non-monotonicity. Referred to as 'n' in the paper.")
parser.add_argument("--device", type=str, choices = ["cpu", "gpu"], default = "gpu", help = "Whether to use cpu or gpu."
                    + "On default falls back to gpu if one exists, falls back to cpu otherwise.")
parser.add_argument("--log", type=float, default=100, help="The logging interval.")
args = parser.parse_args()

def setdevice():
    if args.device == "gpu" and torch.cuda.is_available():
        print("Model will be training on the GPU.\n")
        args.device = torch.device('cuda')
    elif args.device == "gpu":
        print("No GPU detected. Falling back to CPU.\n")
        args.device = torch.device('cpu')
    else:
        print("Model will be training on the CPU.\n")
        args.device = torch.device('cpu')

setdevice()
print('Parameters of the model:')
print('Args:', args)
print("\n")

def save_model(model):
    torch.save({'model_state_dict': model.state_dict()}, args.load)
    
def data_init():
    with open("./data/{}/train.txt".format(args.data), encoding="utf8") as f:
        file = f.read()
        trn = file[1:].split(' ')
    with open("./data/{}/valid.txt".format(args.data), encoding="utf8") as f:
        file = f.read()
        vld = file[1:].split(' ')
    with open("./data/{}/test.txt".format(args.data), encoding="utf8") as f:
        file = f.read()
        tst = file[1:].split(' ')
    words = sorted(set(trn))
    char2ind = {c: i for i, c in enumerate(words)}
    trn = [char2ind[c] for c in trn]
    vld = [char2ind[c] for c in vld]
    tst = [char2ind[c] for c in tst]
    return torch.tensor(trn,dtype=torch.int64).reshape(-1, 1), torch.tensor(vld,dtype=torch.int64).reshape(-1, 1), torch.tensor(tst,dtype=torch.int64).reshape(-1, 1), len(words)

def get_seq_len(bptt):
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

def perplexity(data, model):
    model.eval()
    data = minibatch(data, args.bptt)
    with torch.no_grad():
        losses = []
        batch_size = data[0][0].size(1)
        states = model.state_init(batch_size)
        for x, y in data:
            x = x.to(args.device)
            y = y.to(args.device)
            scores, states = model(x, states)
            loss = F.cross_entropy(scores, y.reshape(-1))
            losses.append(loss.data.item())
    return np.exp(np.mean(losses))

def train(data, model, optimizer):
    trn, vld, tst = data
    tic = timeit.default_timer()
    total_words = 0
    print("Starting finetuning.")
    val_perp = perplexity(vld, model)
    print("Validation perplexity at the start of finetuning process: {:.3f}".format(val_perp))
    best_val = val_perp
    try:
        epoch = 0
        while optimizer.check(val_perp) != True:
            seq_len = get_seq_len(args.bptt)
            num_batch = ((trn.size(0)-1)// seq_len + 1)
            optimizer.lr(seq_len/args.bptt*args.lr)
            states = model.state_init(args.batch_size)
            model.train()
            for i, (x, y) in enumerate(minibatch(trn, seq_len)):
                x = x.to(args.device)
                y = y.to(args.device)
                total_words += x.numel()
                states = model.detach(states)
                scores, states, activations = model(x, states)
                loss = F.cross_entropy(scores, y.reshape(-1))
                h, h_m = activations
                ar_reg = args.ar * h_m.pow(2).mean()
                tar_reg = args.tar * (h[:-1] - h[1:]).pow(2).mean()
                loss_reg = loss + ar_reg + tar_reg
                loss_reg.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if i % (args.log) == 0:
                    toc = timeit.default_timer()
                    print("batch no = {:d} / {:d}, ".format(i, num_batch) +
                          "train loss = {:.3f}, ".format(loss.item()) +
                          "ar val = {:.3f}, ".format(ar_reg.item()) + 
                          "tar val = {:.3f}, ".format(tar_reg.item()) + 
                          "wps = {:d}, ".format(round(total_words/(toc-tic))) +
                          "dw.norm() = {:.3f}, ".format(norm) +
                          "lr = {:.3f}, ".format(seq_len/args.bptt*args.lr) + 
                          "since beginning = {:d} mins, ".format(round((toc-tic)/60)) +
                          "cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated()/1024/1024/1024))
    
            tmp = {}
            for (prm,st) in optimizer.state.items():
                tmp[prm] = prm.clone().detach()
                prm.data = st['ax'].clone().detach()
    
            val_perp = perplexity(vld, model)
    
            if val_perp < best_val:
                best_val = val_perp
                print("Best validation perplexity : {:.3f}".format(best_val))
                save_model(model)
                print("Model saved!")
    
            for (prm, st) in optimizer.state.items():
                prm.data = tmp[prm].clone().detach()
                
            print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, val_perp))
            print("*************************************************\n")
            epoch += 1
    except KeyboardInterrupt:
        print("Finishing finetuning early.")
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['model_state_dict'])
    tst_perp = perplexity(tst, model)
    print("Test set perplexity of best model: {:.3f}".format(tst_perp))

def finetune():
    trn, vld, tst, vocab_size = data_init()
    trn = batchify(trn, args.batch_size)
    vld = batchify(vld, args.valid_batch_size)
    tst = batchify(tst, args.test_batch_size)
    model = Model(vocab_size, args.embed_size, args.hidden_size, args.layer_num, args.w_drop, args.dropout_i, 
                  args.dropout_l, args.dropout_o, args.dropout_e, args.winit, args.lstm_type)
    model.to(args.device)
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = NTASGD(model.parameters(), lr=args.lr, n=args.non_mono, weight_decay=args.weight_decay, fine_tuning=True)
    train((trn, vld, tst), model, optimizer)

finetune()
