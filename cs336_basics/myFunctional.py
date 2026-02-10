import os
import torch
import torch.nn as nn

def toy_softmax(x):
    sx = x -  x.max(-1,keepdim=True).values
    ex = sx.exp()
    return ex/ex.sum(-1,keepdim=True)

def toy_product_atte(Q, K, V, mask=None):
    d_k = torch.tensor(Q.shape[-1])
    Qk = Q @ K.transpose(-2,-1)/ torch.sqrt(d_k)
    if mask != None:
        Qk = Qk.masked_fill(mask==False,float('-inf'))
    sQk = toy_softmax(Qk)
    return sQk @ V

