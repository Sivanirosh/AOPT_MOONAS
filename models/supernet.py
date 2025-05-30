
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


# =========================================
# 3) Primitives
# =========================================
def sep_conv(C, k, s, p):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, k, s, p, groups=C, bias=False),
        nn.Conv2d(C, C, 1, bias=False),
        nn.BatchNorm2d(C),
    )
def dil_conv(C, k, s, p, d):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, k, s, p, dilation=d, bias=False),
        nn.BatchNorm2d(C),
    )
def skip_connect(): return nn.Identity()
def avg_pool(C, k, s, p): return nn.AvgPool2d(k, stride=s, padding=p)

# =========================================
# 4) MixedOp & ResidualCell
# =========================================
class MixedOp(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ops = nn.ModuleList([
            sep_conv(C,3,1,1), sep_conv(C,5,1,2),
            dil_conv(C,3,1,2,2),
            nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C,C,1,bias=False), nn.BatchNorm2d(C)),
            avg_pool(C,3,1,1), skip_connect()
        ])
    def forward(self, x, weights):
        return sum(w*op(x) for w,op in zip(weights,self.ops))

class ResidualCell(nn.Module):
    def __init__(self, C, num_nodes=4):
        super().__init__(); self.num_nodes=num_nodes
        self.pre = nn.Sequential(nn.Conv2d(C,C,1,bias=False), nn.BatchNorm2d(C), nn.ReLU(inplace=False))
        self.ops=nn.ModuleDict({f"{i}<-{j}":MixedOp(C)
                                for i in range(num_nodes+1)
                                for j in range(i+1,num_nodes+1)})
        self.post=nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C*num_nodes,C,1,bias=False), nn.BatchNorm2d(C))
    def forward(self,s0,weights):
        states=[self.pre(s0)]
        for new in range(1,self.num_nodes+1):
            tot=0
            for idx,h in enumerate(states):
                key=f"{idx}<-{new}"
                tot+=self.ops[key](h,weights[key])
            states.append(tot)
        concat=torch.cat(states[1:],dim=1)
        return self.post(concat)+s0

# =========================================
# 5) CellArchParam
# =========================================
class CellArchParam(nn.Module):
    def __init__(self,C,num_nodes=4,num_ops=6):
        super().__init__()
        self.alphas=nn.ParameterDict({f"{i}<-{j}":nn.Parameter(1e-3*torch.randn(num_ops))
                                       for i in range(num_nodes+1)
                                       for j in range(i+1,num_nodes+1)})
    def forward(self,logits):
        return {k:F.softmax(logits[i],dim=-1)
                for i,k in enumerate(self.alphas.keys())}

# =========================================
# SuperNetwork
# =========================================
class SuperNetwork(nn.Module):
    def __init__(self,C=16,num_cells=4,num_nodes=4,num_ops=6,dropout_p=0.2):
        super().__init__(); self.stem=nn.Sequential(nn.Conv2d(3,C,3,2,1,bias=False), nn.BatchNorm2d(C), nn.ReLU(inplace=False), nn.Dropout(dropout_p))
        self.cells=nn.ModuleList([nn.ModuleDict({'arch':CellArchParam(C,num_nodes,num_ops),'body':ResidualCell(C,num_nodes)}) for _ in range(num_cells)])
        self.global_pool=nn.AdaptiveAvgPool2d(1); self.cls_head=nn.Sequential(nn.Dropout(dropout_p),nn.Linear(C,10))
        self.num_blocks=sum(len(c['arch'].alphas) for c in self.cells); self.num_ops=num_ops
    def forward(self,x,arch_logits):
        out=self.stem(x);offset=0
        for cell in self.cells:
            ne=len(cell['arch'].alphas)
            sl=arch_logits[offset:offset+ne];w=cell['arch'](sl)
            out=cell['body'](out,w);offset+=ne
        gap=self.global_pool(out).view(x.size(0),-1);return self.cls_head(gap)