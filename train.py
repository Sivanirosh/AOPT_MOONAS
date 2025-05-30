# train.py

from parsing.arg_parser import ArgParser
from parsing.config_parser import ConfigParser

import os
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from utils.io import get_device, make_reproducible

from data.transforms import get_train_transform, get_val_transform
from data.dataset    import CIFAR10Dataset
from data.dataloader import get_dataloader

from models.factory import get_model_class

from search.updater import MGDAArchUpdater, NSGAIIArchUpdater, RandomSearchArchUpdater
from training.trainer    import Trainer
from training.checkpointer import ModelCheckpointer


def main(args, cfg):
    make_reproducible(cfg.get("seed"))
    device = get_device(args.device_id)

    # ——— Data ———
    train_ds = CIFAR10Dataset(train=True,  transform=get_train_transform())
    val_ds   = CIFAR10Dataset(train=False, transform=get_val_transform())
    train_loader = get_dataloader(
        train_ds,
        batch_size=cfg.get("data", "train", "batch_size"),
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        val_ds,
        batch_size=cfg.get("data", "val", "batch_size"),
        shuffle=False,
        num_workers=args.num_workers
    )

    # ——— Models ———
    snet_cls = get_model_class("supernet")
    hnet_cls = get_model_class("hypernet")

    # instantiate SuperNetwork
    snet = snet_cls(**cfg.get("model", "supernet")).to(device)

    # HyperNetwork needs to know how many blocks & ops in the SuperNetwork
    hcfg = cfg.get("model", "hypernet")
    hnet = hnet_cls(
        snet.num_blocks,
        snet.num_ops,
        **hcfg
    ).to(device)

    # ——— Optimizers ———
    opt_w = torch.optim.SGD(
        snet.parameters(),
        **cfg.get("optimizer", "weight")
    )
    opt_h = torch.optim.SGD(
        hnet.parameters(),
        **cfg.get("optimizer", "arch")
    )

    # ——— Size counts ———
    op_counts = []
    for cell in snet.cells:
        for m in cell["body"].ops.values():
            op_counts.append([sum(p.numel() for p in op.parameters()) for op in m.ops])
    counts_tensor = torch.tensor(op_counts, device=device, dtype=torch.float)
    max_size = counts_tensor.max(dim=1).values.sum().item()

    # ——— Architecture updater selection ———
    strat = cfg.get("search", "strategy")
    if strat == "random":
        upd = RandomSearchArchUpdater(
            optimizer=opt_h,
            sampler=Dirichlet(torch.tensor(cfg.get("search", "alpha"))),
            num_samples=cfg.get("search", "num_samples")
        )
    elif strat == "mgda":
        upd = MGDAArchUpdater(
            optimizer=opt_h,
            mgda_iters=cfg.get("search", "mgda_iters")
        )
    elif strat == "nsga":
        upd = NSGAIIArchUpdater(
            optimizer=opt_h,
            population_size= cfg.get("search", "population_size"),
            num_generations= cfg.get("search", "num_generations"),
            crossover_rate=  cfg.get("search", "crossover_rate"),
            mutation_rate=   cfg.get("search", "mutation_rate"),
            K=               cfg.get("search", "K")
        )
    else:
        raise ValueError(f"Unknown search.strategy: {strat}")

    # ——— Checkpointer & CSV ———
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    csv_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_losses.csv")

    # pass model_kwargs so evaluator can re‐instantiate later
    model_kwargs = {
        "supernet": cfg.get("model", "supernet"),
        "hypernet": cfg.get("model", "hypernet")
    }
    checkpointer = ModelCheckpointer(
        checkpoint_dir  = args.checkpoint_dir,
        experiment_name = args.experiment_name,
        model_kwargs    = model_kwargs
    )

    # if resuming, load epoch & weights
    start_epoch = 0
    if args.resume:
        start_epoch = checkpointer.load_checkpoint(snet, opt_w, device)

    # ——— Trainer ———
    trainer = Trainer(
        supernet     = snet,
        hypernet     = hnet,
        weight_opt   = opt_w,
        arch_opt     = opt_h,
        arch_updater = upd,
        train_loader = train_loader,
        val_loader   = val_loader,
        counts       = counts_tensor,
        max_size     = max_size,
        device       = device,
        csv_path     = csv_path,
        checkpointer = checkpointer,
        start_epoch  = start_epoch,
        early_stop_patience = 10  # early stopping patience, adjust as needed
    )

    trainer.train(args.num_epochs)

    print("Training complete.")


if __name__ == "__main__":
    args = ArgParser().parse_args()
    cfg  = ConfigParser(args.config_file_path)
    main(args, cfg)
