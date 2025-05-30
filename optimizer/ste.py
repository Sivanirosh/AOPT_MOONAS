# optimizer/ste.py

import torch
import torch.nn.functional as F

class ReinMaxSTE(torch.autograd.Function):
    """Straight‐through estimator that discretizes via argmax in forward,
    but uses a soft‐gradient surrogate in backward."""
    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        # Compute softmax and pick the largest class
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        one_hot = F.one_hot(idx, num_classes=logits.size(-1)).type_as(logits)
        ctx.save_for_backward(probs)
        return one_hot

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (probs,) = ctx.saved_tensors
        # d/dlogits of one‐hot ≈ probs*(1−probs) times upstream gradient
        return grad_output * probs * (1 - probs)

# expose a convenient alias
reinforce_max = ReinMaxSTE.apply
