import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        supernet,
        hypernet,
        weight_opt,
        arch_opt,
        arch_updater,
        train_loader,
        val_loader,
        counts,
        max_size,
        device,
        csv_path,
        checkpointer=None,
        start_epoch=0,
        early_stop_patience=10
    ):
        self.snet = supernet
        self.hnet = hypernet
        self.opt_w = weight_opt
        self.opt_h = arch_opt
        self.upd = arch_updater
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.counts = counts
        self.max_size = max_size
        self.device = device
        self.csv_path = csv_path
        self.checkpointer = checkpointer
        self.start_epoch = start_epoch

        self.early_stop_patience = early_stop_patience
        self.best_val_loss = float('inf')
        self.no_improvement_epochs = 0

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")

    def train(self, num_epochs: int):
        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            # 1) Architecture update
            val_batch = next(iter(self.val_loader))
            self.upd.update(
                self.hnet,
                self.snet,
                val_batch,
                self.counts,
                self.max_size,
                self.device
            )

            # 2) Weight update
            total_loss = 0.0
            total_samples = 0
            lam = getattr(
                self.upd,
                'best_lambda',
                torch.tensor([1.0, 0.0], device=self.device)
            )
            iterator = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}",
                leave=False,
                unit="batch"
            )
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                self.opt_w.zero_grad()
                arch_logits = self.hnet(lam)
                out = self.snet(x, arch_logits)
                loss = F.cross_entropy(out, y)
                loss.backward()
                self.opt_w.step()
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                iterator.set_postfix(train_loss=f"{loss.item():.4f}")

            train_loss = total_loss / total_samples

            # 3) Validation loss
            self.snet.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.snet(x, self.hnet(lam))
                    loss = F.cross_entropy(out, y)
                    val_loss += loss.item() * x.size(0)
                    val_samples += x.size(0)
            val_loss /= val_samples
            self.snet.train()

            # Checkpoint if improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improvement_epochs = 0
                if self.checkpointer is not None:
                    self.checkpointer.save(
                        epoch,
                        self.snet,
                        self.hnet,
                        self.opt_w,
                        self.opt_h,
                        val_loss
                    )
            else:
                self.no_improvement_epochs += 1
                print(f"No improvement for {self.no_improvement_epochs} epochs.")

            # Early stopping condition
            if self.no_improvement_epochs >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Log losses
            with open(self.csv_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")