# eval/evaluator.py

import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate(supernet, arch_logits, counts_tensor, max_size, device):
    # Classification accuracy on CIFAR-10 test set
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    cifar_test=datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    loader=DataLoader(cifar_test,batch_size=256,shuffle=False)
    supernet.eval()
    correct,total=0,0
    with torch.no_grad():
        for imgs,labels in loader:
            imgs,labels=imgs.to(device),labels.to(device)
            out=supernet(imgs,arch_logits)
            preds=out.argmax(dim=1)
            correct+= (preds==labels).sum().item(); total+=labels.size(0)
    cls_acc=correct/total
    # Size penalty
    probs=F.softmax(arch_logits,dim=-1); idx=probs.argmax(dim=-1)
    one_hot=F.one_hot(idx,num_classes=arch_logits.size(-1)).to(device).float()
    size_sum=(one_hot*counts_tensor).sum().item()
    size_penalty=size_sum/max_size
    return {'cls_acc':cls_acc,'size_penalty':size_penalty}

def is_pareto_efficient(costs):
    mask=np.ones(costs.shape[0],dtype=bool)
    for i,c in enumerate(costs):
        if mask[i]:
            mask[mask]=np.any(costs[mask]<c,axis=1); mask[i]=True
    return mask