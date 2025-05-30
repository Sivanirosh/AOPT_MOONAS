# optimizer/mgda.py

import torch

def solve_mgda_weights(M: torch.Tensor, iters: int = 50, lr: float = 1e-1, device: torch.device = None) -> torch.Tensor:
    """
    Solve for MGDA weights alpha minimizing alphaᵀ M alpha s.t. alpha>=0, sum alpha=1.
    M: (k,k) Gram matrix of normalized gradients.
    Returns alpha of shape (k,).
    """
    k = M.shape[0]
    alpha = torch.full((k,), 1.0/k, device=device or M.device)
    for _ in range(iters):
        # gradient of alpha^T M alpha = 2 M alpha
        grad = 2 * (M @ alpha)
        alpha = alpha - lr * grad
        # project back to simplex
        u, _ = torch.sort(alpha, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1
        rho = torch.nonzero(u * torch.arange(1, k+1, device=alpha.device) > cssv, as_tuple=False)[-1,0]
        theta = cssv[rho] / (rho.float() + 1)
        alpha = torch.clamp(alpha - theta, min=0.0)
    return alpha
