"""Method 1: contrastive alignment.

Same network as Baseline 1 (we just subclass it), but trained with infonce_loss
instead of mse_cosine_loss. Keeping the architecture identical means any gap
between the two models comes only from the loss function.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.baseline_regression import LatentSpaceRegression


class ContrastiveAlignment(LatentSpaceRegression):
    """Same network as Baseline 1; only the training loss differs (InfoNCE)."""
    pass


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    B = 8
    clip = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    ast = F.normalize(torch.randn(B, config.AST_DIM), dim=-1)
    model = ContrastiveAlignment()
    out = model(clip, ast)
    assert out.shape == (B, config.EMB_DIM), out.shape
    assert torch.allclose(out.norm(dim=-1), torch.ones(B), atol=1e-5), "not unit norm"

    # check the architecture really is identical to Baseline 1
    ref = LatentSpaceRegression()
    a = [(n, tuple(p.shape)) for n, p in model.named_parameters()]
    b = [(n, tuple(p.shape)) for n, p in ref.named_parameters()]
    assert a == b, "ContrastiveAlignment arch differs from LatentSpaceRegression!"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ContrastiveAlignment ok, same arch as Baseline 1, params={n_params:,}")
