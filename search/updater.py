# search/updater.py

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.distributions import Dirichlet

from optimizer.mgda import solve_mgda_weights
from optimizer.nsga2 import *

from optimizer.ste import reinforce_max  


class ArchUpdater(ABC):
    """Base class for architecture‐gradient updaters."""
    @abstractmethod
    def update(
        self,
        hypernet: torch.nn.Module,
        supernet: torch.nn.Module,
        val_batch: Tuple[Tensor, Tensor],
        counts: Tensor,
        max_size: float,
        device: torch.device
    ) -> None:
        """Perform one architecture‐update step."""
        ...


class MGDAArchUpdater(ArchUpdater):
    """MGDA‐based multi‐objective updater."""
    def __init__(self, optimizer: torch.optim.Optimizer, mgda_iters: int = 50):
        self.optimizer = optimizer
        self.mgda_iters = mgda_iters
        self.sampler = Dirichlet(torch.tensor([1.0, 1.0]))

    def update(
        self,
        hypernet: torch.nn.Module,
        supernet: torch.nn.Module,
        val_batch: Tuple[Tensor, Tensor],
        counts: Tensor,
        max_size: float,
        device: torch.device
    ):
        # clear any stale gradients
        self.optimizer.zero_grad()

        x_val, y_val = (t.to(device) for t in val_batch)

        # 1) Sample mixture weights
        lam = self.sampler.rsample().to(device)
        arch_logits = hypernet(lam)
        preds = supernet(x_val, arch_logits)

        # 2) Compute objectives
        loss_cls  = F.cross_entropy(preds, y_val)
        loss_size = (reinforce_max(arch_logits) * counts).sum() / max_size

        # 3) Compute per‐objective gradient vectors
        grads = []
        for loss in (loss_cls, loss_size):
            raw = torch.autograd.grad(
                loss, hypernet.parameters(),
                retain_graph=True, allow_unused=True
            )
            flat = torch.cat([
                (g if g is not None else torch.zeros_like(p)).flatten()
                for g, p in zip(raw, hypernet.parameters())
            ])
            grads.append(flat / (flat.norm() + 1e-8))

        # 4) Solve MGDA subproblem
        G = torch.stack(grads)         # (2, D)
        M = G @ G.T                    # (2, 2)
        alpha = solve_mgda_weights(M, iters=self.mgda_iters, device=device)

        # 5) Combine and scatter back
        combined = alpha[0] * grads[0] + alpha[1] * grads[1]
        offset = 0
        for p in hypernet.parameters():
            n = p.numel()
            p.grad = combined[offset:offset+n].view_as(p)
            offset += n

        # 6) Step
        self.optimizer.step()


class NSGAIIArchUpdater(ArchUpdater):
    """True evolutionary NSGA-II updater."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        population_size: int,
        num_generations: int,
        crossover_rate: float,
        mutation_rate: float,
        K: int = 10
    ):
        self.optimizer = optimizer
        self.pop_size = population_size
        self.num_gens = num_generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.K = K

    def update(
        self,
        hypernet: torch.nn.Module,
        supernet: torch.nn.Module,
        val_batch: Tuple[Tensor, Tensor],
        counts: Tensor,
        max_size: float,
        device: torch.device
    ):
        x_val, y_val = (t.to(device) for t in val_batch)

        # 1) Initialize population of λ’s
        pop = [Dirichlet(torch.tensor([1.0,1.0])).sample().to(device)
               for _ in range(self.pop_size)]

        # 2) Evaluate each individual
        losses = []
        for lam in pop:
            logits = hypernet(lam)
            out = supernet(x_val, logits)
            lc = F.cross_entropy(out, y_val).item()
            ls = ((reinforce_max(logits) * counts).sum() / max_size).item()
            losses.append((lc, ls))

        # 3) Do NSGA-II selection + variation (crossover/mutation)
        #    (you can reuse your select_by_nsga2 to pick the first front
        #     + crowding for survivors, then apply simple SBX + Gaussian mut)
        new_pop = evolve_nsga2(
            pop, losses,
            pop_size=self.pop_size,
            crossover_rate=self.cx_rate,
            mutation_rate=self.mut_rate
        )

        # 4) From the final front, pick the top‐K by crowding distance
        final_losses = [(F.cross_entropy(supernet(x_val, hypernet(l)), y_val).item(),
                         ((reinforce_max(hypernet(l)) * counts).sum() / max_size).item())
                        for l in new_pop]
        final_grads = []
        for lam in new_pop:
            self.optimizer.zero_grad()
            logits = hypernet(lam)
            loss = F.cross_entropy(supernet(x_val, logits), y_val)
            loss.backward()
            final_grads.append(torch.cat([p.grad.flatten() for p in hypernet.parameters()]))
        avg_grad = select_by_nsga2(final_losses, final_grads, self.K)

        # 5) Scatter & step
        offset = 0
        self.optimizer.zero_grad()
        for p in hypernet.parameters():
            n = p.numel()
            p.grad = avg_grad[offset:offset + n].view_as(p)
            offset += n
        self.optimizer.step()


class RandomSearchArchUpdater(ArchUpdater):
    """Random-search over λ; final step on best λ."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        sampler: Dirichlet,
        num_samples: int = 10
    ):
        self.optimizer = optimizer
        self.sampler = sampler
        self.num_samples = num_samples
        self.best_lambda = torch.tensor([1.0, 0.0])

    def update(
        self,
        hypernet: torch.nn.Module,
        supernet: torch.nn.Module,
        val_batch: Tuple[Tensor, Tensor],
        counts: Tensor,
        max_size: float,
        device: torch.device
    ):
        x_val, y_val = (t.to(device) for t in val_batch)

        best_score = float('inf')
        for _ in range(self.num_samples):
            lam = self.sampler.rsample().to(device)
            arch_logits = hypernet(lam)
            out = supernet(x_val, arch_logits)

            lc = F.cross_entropy(out, y_val).item()
            ls = ((reinforce_max(arch_logits) * counts).sum() / max_size).item()
            score = lam[0]*lc + lam[1]*ls

            if score < best_score:
                best_score = score
                self.best_lambda = lam.detach()

        # final gradient step on best λ
        self.optimizer.zero_grad()
        final_logits = hypernet(self.best_lambda)
        loss = F.cross_entropy(supernet(x_val, final_logits), y_val)
        loss.backward()
        self.optimizer.step()
