import torch
import torch.nn as nn
import os

def toy_softmax(x):
    sx = x -  x.max(-1,keepdim=True).values
    ex = sx.exp()
    return ex/ex.sum(-1,keepdim=True)

def toy_product_atte(Q, K, V, mask=None):
    d_k = torch.tensor(Q.shape[-1])
    Qk = Q @ K.transpose(-2,-1)/ torch.sqrt(d_k)
    if mask is not None:
        Qk = Qk.masked_fill(mask==False,float('-inf'))
    sQk = toy_softmax(Qk)
    return sQk @ V

def toy_cross_entry(logits, targets):
    #loss = -exp(logits)/sum(exp(logits))[pi].log().mean()
    #     = -(logits[pi] - sum(exp(logits).log)).mean()
    m_logits = logits - logits.max(-1,keepdim=True).values
    log_probs =m_logits - m_logits.exp().sum(-1,keepdim = True).log()
    t = torch.arange(len(targets))
    return -(log_probs[t,targets]).mean()

from math import cos,pi
def cosine_warm_up_lr(    
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,):
    
    lr_now = 0
    if it < warmup_iters:
        lr_now = it * max_learning_rate / warmup_iters
    elif it <= cosine_cycle_iters:
        lr_now = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + cos((it-warmup_iters)*pi/(cosine_cycle_iters-warmup_iters))) 
    else:
        lr_now = min_learning_rate
        
    return lr_now

@torch.no_grad()
def toy_grad_clip(parameters,max_l2_norm):
    norm_all = torch.zeros(1)
    for p in parameters:
        if p.grad is not None:
            norm_all += p.grad.norm().square()
    norm_all = norm_all.sqrt()
    if norm_all > max_l2_norm:
        scale = max_l2_norm / norm_all
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)
    return
            