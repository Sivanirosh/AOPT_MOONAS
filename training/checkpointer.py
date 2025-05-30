# training/checkpointer.py

import os
import torch

class ModelCheckpointer:
    """
    Saves and loads both the SuperNetwork and HyperNetwork
    (plus optimizers and model_kwargs) into a single .pt file.
    """
    def __init__(self, checkpoint_dir: str, experiment_name: str, model_kwargs: dict):
        self.ckpt_dir = checkpoint_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.filename = f"{experiment_name}.pt"
        self.best_loss = float('inf')
        self.model_kwargs = model_kwargs

    def save(
        self,
        epoch: int,
        supernet: torch.nn.Module,
        hypernet: torch.nn.Module,
        opt_w: torch.optim.Optimizer,
        opt_h: torch.optim.Optimizer,
        val_loss: float
    ):
        """
        Save if val_loss improves. Bundles:
          - epoch
          - state_dicts of both nets
          - state_dicts of both optimizers
          - the model_kwargs dict
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            path = os.path.join(self.ckpt_dir, self.filename)
            torch.save({
                'epoch': epoch,
                'supernet': supernet.state_dict(),
                'hypernet': hypernet.state_dict(),
                'opt_w_state': opt_w.state_dict(),
                'opt_h_state': opt_h.state_dict(),
                'model_kwargs': self.model_kwargs,
            }, path)

    def load_checkpoint(
        self,
        supernet: torch.nn.Module,
        hypernet: torch.nn.Module,
        opt_w: torch.optim.Optimizer,
        opt_h: torch.optim.Optimizer,
        device: torch.device
    ) -> int:
        """
        If a checkpoint exists, load both nets and both optimizers.
        Returns the last saved epoch, or 0 if none.
        """
        path = os.path.join(self.ckpt_dir, self.filename)
        if not os.path.exists(path):
            return 0
        ckpt = torch.load(path, map_location=device)
        supernet.load_state_dict(ckpt['supernet'])
        hypernet.load_state_dict(ckpt['hypernet'])
        opt_w.load_state_dict(ckpt['opt_w_state'])
        opt_h.load_state_dict(ckpt['opt_h_state'])
        return ckpt.get('epoch', 0)
