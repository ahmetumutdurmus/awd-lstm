import torch
import torch.optim as optim

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