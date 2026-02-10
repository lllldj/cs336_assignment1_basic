import torch
import torch.nn as nn
import os


class toy_Liner(nn.Module):
    def __init__(self, in_features, out_features, bias=None, device=None, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features,dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features,dtype=dtype)) if bias else None
        self.device = device
        
        self.set_weights()
    
    def set_weights(self,w=None):
        if w == None:
            nn.init.trunc_normal_(self.weight)
        else:
            self.weight.data = w
    
    def forward(self,x):
        out = x @ self.weight.T
        if self.bias != None:
            out += self.bias
        return out
    

class toy_Embedding(nn.Module):
    def __init__(self, num_embd, embd_dim, device = None,dtype = torch.float32) -> None:
        super().__init__()
        self.embd = nn.Parameter(torch.empty(num_embd,embd_dim,dtype=dtype))
        self.device = device

        self.set_embd()
        
    def set_embd(self,embd=None):
        if embd == None:
            nn.init.trunc_normal_(self.embd)
        else:
            self.embd.data = embd 
    
    def forward(self,x):
        return self.embd[x]
    
class toy_RMSnorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-5, device = None, dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.empty(d_model,dtype=dtype))

        self.set_para()
        
    def set_para(self,g=None):
        if g==None:
            nn.init.trunc_normal_(self.gain,1,0.02)
        else:
            self.gain.data = g
    
    def forward(self,x):
        rmsx = x.square().mean(-1,keepdim=True) 
        out = x*self.gain/torch.sqrt(rmsx+self.eps)
        return out 
    
class toy_SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=torch.float32):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_ff,d_model,dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model,d_ff,dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff,d_model,dtype=dtype))
    
        self.set_para()
    def set_para(self,w1=None,w2=None,w3=None):
        if w1 == None:
            nn.init.trunc_normal_(self.W1)
        else:
            self.W1.data = w1
        if w2 == None:
            nn.init.trunc_normal_(self.W2)
        else:
            self.W2.data = w2
        if w3 == None:
            nn.init.trunc_normal_(self.W3)
        else:
            self.W3.data = w3
    
    def forward(self,x):
        W3x = x @ self.W3.T
        W1x = x @ self.W1.T
        Slu = W1x * torch.sigmoid(W1x)
        return (Slu * W3x)@self.W2.T
    
    
class toy_RoPE(nn.Module):
    def __init__(self, d_k, theta, max_len, device = None, dtype = torch.float32):
        super().__init__()
        
        self.rot_d = d_k//2
        i = torch.arange(self.rot_d, device=device, dtype=dtype)         
        j = torch.arange(max_len, device=device, dtype=dtype)      

        inv_freq = torch.exp(-(2*i)/d_k * torch.log(torch.tensor(theta, device=device, dtype=dtype)))                   
        thetas = j[:, None] * inv_freq[None, :]  
        
        cos_table = torch.cos(thetas)  #cos_table [token posistion, feature posistion]
        sin_table = torch.sin(thetas)
        
        self.register_buffer("cos_table",cos_table,persistent=False)
        self.register_buffer("sin_table",sin_table,persistent=False)
    
    def forward(self,x,tk_posistions):
        cos = self.cos_table[tk_posistions] #(T,d/2)
        sin = self.sin_table[tk_posistions] #(T,d/2)
        x_rot = x[..., :2*self.rot_d]
        x_pass = x[..., 2*self.rot_d:]
        x1 = x_rot[...,0::2] #(T,d/2 + 1) ?
        x2 = x_rot[...,1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        y_rot = torch.stack([y1, y2], dim=-1).flatten(-2)
        return torch.cat([y_rot, x_pass], dim=-1)