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
# 7) HyperNetwork
# =========================================
class HyperNetwork(nn.Module):
    def __init__(self,num_blocks,num_ops,hidden_dim=128,depth=3):
        super().__init__(); layers=[nn.Linear(2,hidden_dim)]
        for _ in range(depth-1): layers+= [nn.ReLU(inplace=False),nn.Linear(hidden_dim,hidden_dim),nn.LayerNorm(hidden_dim)]
        self.mlp=nn.Sequential(*layers); self.fc_out=nn.Linear(hidden_dim,num_blocks*num_ops)
        self.num_blocks=num_blocks; self.num_ops=num_ops
    def forward(self,lambda_vec):
        x=lambda_vec.unsqueeze(0) if lambda_vec.dim()==1 else lambda_vec
        h=self.mlp(x);out=self.fc_out(h).view(-1,self.num_blocks,self.num_ops)
        return out.squeeze(0) if lambda_vec.dim()==1 else out