import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.factory import get_model_class
from data.transforms import get_train_transform, get_val_transform
from data.dataset import CIFAR10Dataset
from data.dataloader import get_dataloader

# 1) Define primitives once (must match SuperNetwork)
from models.supernet import sep_conv, dil_conv, skip_connect, avg_pool

_PRIMITIVES = [
    lambda C: sep_conv(C,3,1,1),
    lambda C: sep_conv(C,5,1,2),
    lambda C: dil_conv(C,3,1,2,2),
    lambda C: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, 1, bias=False),
        nn.BatchNorm2d(C)
    ),
    lambda C: avg_pool(C,3,1,1),
    lambda C: skip_connect()
]

class DiscreteCell(nn.Module):
    def __init__(self, C, num_nodes, choices):
        super().__init__()
        self.num_nodes = num_nodes
        self.pre = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False)
        )
        self.ops = nn.ModuleDict({
            key: _PRIMITIVES[idx](C)
            for key, idx in choices.items()
        })
        self.post = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C * num_nodes, C, 1, bias=False),
            nn.BatchNorm2d(C)
        )

    def forward(self, x):
        states = [self.pre(x)]
        for new in range(1, self.num_nodes + 1):
            out_sum = 0
            for i, h in enumerate(states):
                key = f"{i}<-{new}"
                out_sum += self.ops[key](h)
            states.append(out_sum)
        concat = torch.cat(states[1:], dim=1)
        return self.post(concat) + x

class DiscreteSuperNetwork(nn.Module):
    def __init__(self, C, num_cells, num_nodes, num_ops, dropout_p=0.2, discrete_arch=None):
        super().__init__()
        self.C = C
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_p)
        )
        edges_per_cell = (num_nodes + 1) * num_nodes // 2
        assert discrete_arch is not None and len(discrete_arch) == num_cells * edges_per_cell
        ptr = 0
        self.cells = nn.ModuleList()
        for _ in range(num_cells):
            cell_choices = {}
            for i in range(num_nodes + 1):
                for j in range(i+1, num_nodes + 1):
                    key = f"{i}<-{j}"
                    cell_choices[key] = discrete_arch[ptr]
                    ptr += 1
            self.cells.append(DiscreteCell(C, num_nodes, cell_choices))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(C, 10)
        )

    def forward(self, x):
        out = self.stem(x)
        for cell in self.cells:
            out = cell(out)
        gap = self.global_pool(out).view(out.size(0), -1)
        return self.cls_head(gap)


def retrain_fixed_arch(
    discrete_arch: list,
    model_kwargs: dict,
    epochs: int,
    device: torch.device,
    output_dir: str,
    prefix: str
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{prefix}_losses.csv")
    with open(csv_path, 'w') as cf:
        cf.write('epoch,train_loss,val_loss\n')

    # recompute counts with dummy supernet
    from models.supernet import SuperNetwork
    dummy = SuperNetwork(**model_kwargs).to(device)
    op_counts = []
    for cell in dummy.cells:
        for mixed in cell['body'].ops.values():
            op_counts.append([sum(p.numel() for p in op.parameters()) for op in mixed.ops])
    counts_tensor = torch.tensor(op_counts, device=device, dtype=torch.float)
    max_size = counts_tensor.max(1).values.sum().item()

    dsnet = DiscreteSuperNetwork(discrete_arch=discrete_arch, **model_kwargs).to(device)
    train_ds = CIFAR10Dataset(train=True, transform=get_train_transform())
    val_ds   = CIFAR10Dataset(train=False, transform=get_val_transform())
    train_loader = get_dataloader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader   = get_dataloader(val_ds,   batch_size=128, shuffle=False, num_workers=4)
    optimizer = torch.optim.SGD(dsnet.parameters(), lr=0.025, momentum=0.9)
    best_val = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # TRAIN EPOCH
        dsnet.train()
        total, count = 0.0, 0
        iterator = tqdm(train_loader, desc=f"Retrain {prefix} Epoch {epoch}", leave=False, unit="batch")
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(dsnet(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * y.size(0)
            count += y.size(0)
            iterator.set_postfix(train_loss=f"{loss.item():.4f}")
        train_loss = total / count
        train_losses.append(train_loss)

        # VAL EPOCH
        dsnet.eval()
        total_v, count_v = 0.0, 0
        iterator_v = tqdm(val_loader, desc=f"Validate {prefix} Epoch {epoch}", leave=False, unit="batch")
        with torch.no_grad():
            for x, y in iterator_v:
                x, y = x.to(device), y.to(device)
                loss_v = F.cross_entropy(dsnet(x), y)
                total_v += loss_v.item() * y.size(0)
                count_v += y.size(0)
                iterator_v.set_postfix(val_loss=f"{loss_v.item():.4f}")
        val_loss = total_v / count_v
        val_losses.append(val_loss)

        # checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(dsnet.state_dict(), os.path.join(output_dir, f"{prefix}_best.pth"))

        # log CSV
        with open(csv_path, 'a') as cf:
            cf.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")

    # TEST
    dsnet.load_state_dict(torch.load(os.path.join(output_dir, f"{prefix}_best.pth")))
    dsnet.eval()
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    correct, total_t = 0, 0
    for x, y in tqdm(test_loader, desc=f"Test {prefix}", leave=False, unit="batch"):
        x, y = x.to(device), y.to(device)
        preds = dsnet(x).argmax(dim=1)
        correct += (preds==y).sum().item()
        total_t += y.size(0)
    final_acc = correct / total_t

    # size penalty
    arch_idx = torch.tensor(discrete_arch, device=device, dtype=torch.long)
    one_hot = F.one_hot(arch_idx, num_classes=model_kwargs['num_ops']).float()
    size_sum = (one_hot * counts_tensor).sum().item()
    size_penalty = size_sum / max_size

    return {
        'final_acc': final_acc,
        'size_penalty': size_penalty,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'loss_csv': csv_path,
        'best_checkpoint': os.path.join(output_dir, f"{prefix}_best.pth")
    }
