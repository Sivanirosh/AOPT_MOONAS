import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torchvision import datasets, transforms

from parsing.config_parser import ConfigParser
from models.factory import get_model_class
from optimizer.ste import reinforce_max  # straight-through estimator
from eval.evaluator import is_pareto_efficient
from training.retrainer import retrain_fixed_arch

# --------------------------------------------------------------------
# 1) Command-line interface
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Evaluate and retrain architectures from supernet checkpoints."
)
parser.add_argument("--device-id",         type=int,   required=True)
parser.add_argument("--config-file-path",  type=str,   required=True)
parser.add_argument("--mgda-checkpoint",   type=str,   required=True)
parser.add_argument("--nsga-checkpoint",   type=str,   required=True)
parser.add_argument("--random-checkpoint", type=str,   required=True)
parser.add_argument("--n-samples",         type=int,   default=20)
parser.add_argument("--output-dir",        type=str,   required=True)
parser.add_argument("--retrain-epochs",    type=int,   default=30)
args = parser.parse_args()

# Set up device and base output
os.makedirs(args.output_dir, exist_ok=True)
plots_dir = os.path.join(args.output_dir, 'plots')
models_dir = os.path.join(args.output_dir, 'models')
losses_dir = os.path.join(args.output_dir, 'losses')
for d in (plots_dir, models_dir, losses_dir):
    os.makedirs(d, exist_ok=True)

device = torch.device(
    f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------------------------
# 2) Load configuration
# --------------------------------------------------------------------
cfg = ConfigParser(args.config_file_path)
model_cfg = cfg.get('model')

# --------------------------------------------------------------------
# 3) Helpers
# --------------------------------------------------------------------
def load_models(ckpt_path):
    chk = torch.load(ckpt_path, map_location=device)
    assert 'model_kwargs' in chk, f"Checkpoint {ckpt_path} missing 'model_kwargs'"
    super_kwargs = chk['model_kwargs']['supernet']
    hyper_kwargs = chk['model_kwargs']['hypernet']
    snet_cls = get_model_class('supernet')
    hnet_cls = get_model_class('hypernet')
    snet = snet_cls(**super_kwargs).to(device)
    hnet = hnet_cls(snet.num_blocks, snet.num_ops, **hyper_kwargs).to(device)
    snet.load_state_dict(chk['supernet'])
    hnet.load_state_dict(chk['hypernet'])
    snet.eval()
    return snet, hnet


def get_test_loader(batch_size=256):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    ds = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=tf
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

# --------------------------------------------------------------------
# 4) Sampling & combined Pareto
# --------------------------------------------------------------------
strategies = {
    'mgda':   args.mgda_checkpoint,
    'nsga':   args.nsga_checkpoint,
    'random': args.random_checkpoint
}
all_metrics = {}
all_masks = {}
all_lambdas = {}

for name, ckpt in strategies.items():
    print(f"____ Sampling {name} ____")
    snet, hnet = load_models(ckpt)

    # recompute counts & max_size
    op_counts = []
    for cell in snet.cells:
        for mixed in cell.body.ops.values():
            op_counts.append([
                sum(p.numel() for p in op.parameters())
                for op in mixed.ops
            ])
    counts = torch.tensor(op_counts, device=device, dtype=torch.float)
    max_size = counts.max(1).values.sum().item()

    sampler = Dirichlet(torch.ones(2, device=device))
    metrics = np.zeros((args.n_samples, 2))  # [size_penalty, accuracy]
    lambdas = []
    loader = get_test_loader()

    for i in range(args.n_samples):
        lam = sampler.rsample()
        lambdas.append(lam.cpu().tolist())
        # accuracy
        total = correct = 0
        with torch.no_grad():
            logits = hnet(lam)
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                pred = snet(x, logits).argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        acc = correct/total
        # size penalty
        probs = F.softmax(logits, -1)
        one_hot = F.one_hot(probs.argmax(-1), probs.size(-1)).float().to(device)
        size_pen = (one_hot * counts).sum().item() / max_size
        metrics[i] = (size_pen, acc)

    # pareto mask on (size, -acc)
    costs = np.stack([metrics[:,0], -metrics[:,1]], axis=1)
    mask = is_pareto_efficient(costs)

    all_metrics[name] = metrics
    all_masks[name]   = mask
    all_lambdas[name] = lambdas

# plot approximated Pareto
plt.figure()
colors = {'mgda':'r','nsga':'g','random':'b'}
labels = {'mgda':'MGDA','nsga':'NSGA','random':'Random'}
for name in strategies:
    m = all_metrics[name]
    mask = all_masks[name]
    # scatter all samples
    plt.scatter(m[:,0], m[:,1], color=colors[name], alpha=0.3, label=f"{labels[name]} samples")
    # get front points and sort by size for line
    front = m[mask]
    order = np.argsort(front[:,0])
    front = front[order]
    plt.plot(front[:,0], front[:,1], linestyle='--', color=colors[name],
             linewidth=2, label=f"{labels[name]} front")

plt.xlabel('Size penalty')
plt.ylabel('Accuracy')
plt.title('Approximated Pareto across strategies')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'approximated_pareto.png'))
plt.close()

# --------------------------------------------------------------------
# 5) Select best lambdas and discrete architectures
# --------------------------------------------------------------------
print("_______________________")
best_archs = {}
for name in strategies:
    print(f"___ Selecting best arch for {name} ___")
    metrics = all_metrics[name]
    mask = all_masks[name]
    idxs = np.where(mask)[0] if mask.any() else np.arange(len(metrics))
    best_i = idxs[np.argmax(metrics[idxs,1])]
    lam = torch.tensor(all_lambdas[name][best_i], device=device)
    snet, hnet = load_models(strategies[name])
    with torch.no_grad():
        arch_logits = hnet(lam)
        discrete = F.softmax(arch_logits, -1).argmax(-1).cpu().tolist()
    best_archs[name] = discrete
    yaml_path = os.path.join(models_dir, f'{name}_best.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({'discrete_arch': discrete}, f)

# --------------------------------------------------------------------
# 6) Retraining
# --------------------------------------------------------------------
print("_______________________")
print("___ Retraining selected architectures ___")
finals = {}
for name, arch in best_archs.items():
    print(f"Retraining {name}...")
    out = retrain_fixed_arch(
        discrete_arch=arch,
        model_kwargs=model_cfg['supernet'],
        epochs=args.retrain_epochs,
        device=device,
        output_dir=os.path.join(losses_dir, name),
        prefix=name
    )
    os.replace(out['best_checkpoint'], os.path.join(models_dir, f'{name}_best.pth'))
    # os.replace(out['loss_csv'], os.path.join(losses_dir, f'{name}_losses.csv'))
    finals[name] = (out['size_penalty'], out['final_acc'])

# plot True Pareto
plt.figure()
for name, (size_pen, acc) in finals.items():
    plt.scatter(size_pen, acc, marker='o', s=100, label=name.upper())
plt.xlabel('Size penalty')
plt.ylabel('Accuracy')
plt.title('True Pareto after retraining')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'true_pareto.png'))
plt.close()

# --------------------------------------------------------------------
# 7) Qualitative visualization of best models
# --------------------------------------------------------------------
from data.transforms import get_val_transform  # ensure imports at top
from data.dataset import CIFAR10Dataset
from torch.utils.data import DataLoader

# prepare small validation loader
val_dataset = CIFAR10Dataset(train=False, transform=get_val_transform())
vis_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

print("_______________________")
for name, arch in best_archs.items():
    print(f"Visualizing {name} architecture...")
    from training.retrainer import DiscreteSuperNetwork
    net = DiscreteSuperNetwork(
        **model_cfg['supernet'],
        discrete_arch=arch
    ).to(device)
    # load retrained weights
    ckpt = torch.load(os.path.join(models_dir, f"{name}_best.pth"), map_location=device)
    net.load_state_dict(ckpt)
    net.eval()

    imgs, labels = next(iter(vis_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    with torch.no_grad():
        logits = net(imgs)
        preds  = logits.argmax(dim=1)

    # unnormalize for display
    unnorm = transforms.Normalize(
        mean=[-m/s for m,s in zip((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))],
        std=[1/s for s in (0.2470,0.2435,0.2616)]
    )
    imgs_disp = torch.clamp(unnorm(imgs), 0, 1)

    plt.figure(figsize=(12,4))
    plt.suptitle(f"{name.upper()} sample predictions", fontsize=16)
    for i in range(imgs_disp.size(0)):
        ax = plt.subplot(2, 4, i+1)
        img_np = imgs_disp[i].cpu().permute(1,2,0).numpy()
        ax.imshow(img_np)
        true_lbl = val_dataset.classes[labels[i].item()]
        pred_lbl = val_dataset.classes[preds[i].item()]
        color = "green" if preds[i]==labels[i] else "red"
        ax.set_title(f"T: {true_lbl}, P: {pred_lbl}", color=color)
        ax.axis('off')
    plt.tight_layout(rect=[0,0,1,0.9])
    plt.savefig(os.path.join(plots_dir, f"{name}_vis.png"))
    plt.close()

print("_______________________")
print("Done. See outputs in", args.output_dir)
