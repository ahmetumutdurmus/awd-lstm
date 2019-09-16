import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

class Embed(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.W = nn.Parameter(torch.Tensor(vocab_size, embed_size))

    def mask(self):
        mask = torch.empty(self.vocab_size, 1, device = self.W.device).bernoulli_(1-self.dropout)/(1-self.dropout)
        return mask.expand(self.vocab_size, self.embed_size)
                     
    def forward(self, x):
        if self.training:
            W = self.mask() * self.W 
            return W[x]
        else:
            return self.W[x]

    def __repr__(self):
        return "Embedding(vocab: {}, embedding: {}, dropout: {})".format(self.vocab_size, self.embed_size, self.dropout)

class VariationalDropout(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, dropout):
        if self.training:
            mask = torch.empty(input.size(1), input.size(2), device = input.device).bernoulli_(1-dropout)/(1-dropout)
            mask = mask.expand_as(input)
            return mask * input
        else:
            return input
    
    def __repr__(self):
        return "VariationalDropout()"
    
class WeightDropLSTM(nn.Module):
    def __init__(self, input, hidden, weight_drop = 0.0):
        super().__init__()
        self.input_size = input
        self.hidden_size = hidden
        self.weight_drop = weight_drop
        self.module = nn.LSTM(input, hidden)
        self.weight_name = 'weight_hh_l0'
        w = getattr(self.module, self.weight_name)
#        self.register_parameter(f'{self.weight_name}_raw', nn.Parameter(w.data))
        self.register_parameter(f'{self.weight_name}_raw', nn.Parameter(w.clone().detach()))
        raw_w = getattr(self, f'{self.weight_name}_raw')
        self.module._parameters[self.weight_name] = F.dropout(raw_w, p=self.weight_drop, training=False)
#        self.module._parameters[self.weight_name] = F.dropout(w, p=self.weight_drop, training=False)
#        self.module.flatten_parameters()

    def __repr__(self):
        return "WeightDropLSTM(input: {}, hidden: {}, weight drop: {})".format(self.input_size, self.hidden_size, self.weight_drop)

    def _setweights(self):
        raw_w = getattr(self, f'{self.weight_name}_raw')
        self.module._parameters[self.weight_name] = F.dropout(raw_w, p=self.weight_drop, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

class WeightDropLSTMCustom(nn.Module):
    def __init__(self, input, hidden, dropout = 0):
        super().__init__()
        self.input_size = input
        self.hidden_size = hidden
        self.dropout = dropout
        self.W_x = nn.Parameter(torch.Tensor(4 * hidden, input))
        self.W_h = nn.Parameter(torch.Tensor(4 * hidden, hidden))
        self.b_x = nn.Parameter(torch.Tensor(4 * hidden))
        self.b_h = nn.Parameter(torch.Tensor(4 * hidden))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.W_x, -stdv, stdv)
        nn.init.uniform_(self.W_h, -stdv, stdv)
        nn.init.uniform_(self.b_x, -stdv, stdv)
        nn.init.uniform_(self.b_h, -stdv, stdv)

    def __repr__(self):
        return "WeightDropLSTM(input: {}, hidden: {}, dropout: {})".format(self.input_size, self.hidden_size, self.dropout)

    def lstm_step(self, x, h, c, W_x, W_h, b_x, b_h):
        gx = torch.addmm(b_x, x, W_x.t())
        gh = torch.addmm(b_h, h, W_h.t())
        xi, xf, xo, xn = gx.chunk(4, 1)
        hi, hf, ho, hn = gh.chunk(4, 1)
        inputgate = torch.sigmoid(xi + hi)
        forgetgate = torch.sigmoid(xf + hf)
        outputgate = torch.sigmoid(xo + ho)
        newgate = torch.tanh(xn + hn)
        c = forgetgate * c + inputgate * newgate
        h = outputgate * torch.tanh(c)
        return h, c

    #Takes input tensor x with dimensions: [T, B, X].
    def forward(self, input, states):
        h, c = states
        outputs = []
        inputs = input.unbind(0)
        W_h = F.dropout(self.W_h, self.dropout, self.training)
        for x_t in inputs:
            h, c = self.lstm_step(x_t, h, c, self.W_x, W_h, self.b_x, self.b_h)
            outputs.append(h)
        return torch.stack(outputs), (h, c)
    
class FC_tied(nn.Module):
    def __init__(self, input, hidden, weight):
        super().__init__()
        self.input_size = input
        self.hidden_size = hidden
        self.W = weight
        self.b = nn.Parameter(torch.Tensor(hidden))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.b, -stdv, stdv)

    def forward(self, x):
        z = torch.addmm(self.b, x.view(-1, x.size(2)), self.W.t())
        return z

    def __repr__(self):
        return "FC(input: {}, hidden: {})".format(self.input_size, self.hidden_size)

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = Embed(config.vocab_size, config.embed_size, config.dropout_embed)
        self.lstm1 = WeightDropLSTM(config.embed_size, config.hidden_size, config.weight_drop)
        self.lstm2 = WeightDropLSTM(config.hidden_size, config.hidden_size, config.weight_drop)
        self.lstm3 = WeightDropLSTM(config.hidden_size, config.embed_size, config.weight_drop)
        self.fc = FC_tied(config.embed_size, config.vocab_size, self.embed.W)
        self.dropout_layer = VariationalDropout()
        self.dropout_i = config.dropout_i
        self.dropout_l = config.dropout_l
        self.dropout_y = config.dropout_y
        self.init_e = config.init_e
        self.weight_init()
        
    def forward(self, input, states):
        s1, s2, s3 = states
        word_vec = self.embed(input)
        word_vec = self.dropout_layer(word_vec, self.dropout_i)
        h1, s1 = self.lstm1(word_vec, s1)
        h1 = self.dropout_layer(h1, self.dropout_l)
        h2, s2 = self.lstm2(h1, s2)
        h2 = self.dropout_layer(h2, self.dropout_l)
        h3, s3 = self.lstm3(h2, s3)
        h3_masked = self.dropout_layer(h3, self.dropout_y)
        scores = self.fc(h3_masked)
        if self.training:
            return scores, (s1, s2, s3), (h3, h3_masked)
        else:
            return scores, (s1, s2, s3)
    
    def weight_init(self):
        nn.init.uniform_(self.embed.W, -self.init_e, self.init_e)
        
    def state_init(self, batch_size):
        dev = next(self.parameters()).device
        s1 = torch.zeros(1, batch_size, self.lstm1.hidden_size, device = dev), torch.zeros(1, batch_size, self.lstm1.hidden_size, device = dev)
        s2 = torch.zeros(1, batch_size, self.lstm2.hidden_size, device = dev), torch.zeros(1, batch_size, self.lstm2.hidden_size, device = dev)
        s3 = torch.zeros(1, batch_size, self.lstm3.hidden_size, device = dev), torch.zeros(1, batch_size, self.lstm3.hidden_size, device = dev)
        return s1, s2, s3
    
    def detach(self, states):
        s1, s2, s3 = states
        return (s1[0].detach(), s1[1].detach()), (s2[0].detach(), s2[1].detach()), (s3[0].detach(), s3[1].detach())
