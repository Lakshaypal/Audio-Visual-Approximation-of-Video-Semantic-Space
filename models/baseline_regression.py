"""Baseline 1: latent-space regression.

A plain MLP that concatenates the cached CLIP and AST embeddings and regresses
the S3D teacher embedding. Trained with mse_cosine_loss. forward returns an
L2-normalized (B, 512) vector. Method 1 reuses this exact network, so the only
thing different between them is the loss.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LatentSpaceRegression(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = config.EMB_DIM + config.AST_DIM        # 512 + 768 = 1280
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, config.EMB_DIM),            # 1024 -> 512
            nn.LayerNorm(config.EMB_DIM),
            nn.ReLU(),
            nn.Linear(config.EMB_DIM, config.EMB_DIM),  # 512 -> 512
        )

    def forward(self, clip_emb: torch.Tensor, ast_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([clip_emb, ast_emb], dim=-1)      # (B, 1280)
        return F.normalize(self.net(x), p=2, dim=-1)    # (B, 512), unit norm


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    B = 8
    clip = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    ast = F.normalize(torch.randn(B, config.AST_DIM), dim=-1)
    model = LatentSpaceRegression()
    out = model(clip, ast)
    n_params = sum(p.numel() for p in model.parameters())
    assert out.shape == (B, config.EMB_DIM), out.shape
    assert torch.allclose(out.norm(dim=-1), torch.ones(B), atol=1e-5), "not unit norm"
    print(f"LatentSpaceRegression ok, out {tuple(out.shape)}, params={n_params:,}")
