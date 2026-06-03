"""Method 2: cross-modal distillation.

Projects the AST audio vector into the CLIP space, runs cross-attention both
ways between audio and image, concatenates the two outputs and projects back to
512. Trained with crd_loss against the S3D teacher.

We cache one pooled vector per modality (not token sequences), so each modality
goes into attention as a length-1 sequence of shape (B, 1, 512). The same code
would still work if we ever cached longer token sequences.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

NUM_HEADS = 8


class CrossModalDistillation(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_proj = nn.Linear(config.AST_DIM, config.EMB_DIM)   # 768 -> 512
        self.attn1 = nn.MultiheadAttention(
            embed_dim=config.EMB_DIM, num_heads=NUM_HEADS, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            embed_dim=config.EMB_DIM, num_heads=NUM_HEADS, batch_first=True
        )
        self.fusion = nn.Linear(2 * config.EMB_DIM, config.EMB_DIM)   # 1024 -> 512

    def forward(self, clip_emb: torch.Tensor, ast_emb: torch.Tensor) -> torch.Tensor:
        ast_proj = self.audio_proj(ast_emb)            # (B, 512)
        clip_seq = clip_emb.unsqueeze(1)               # (B, 1, 512)
        ast_seq = ast_proj.unsqueeze(1)                # (B, 1, 512)

        # attn1: audio queries image; attn2: image queries audio
        o1, _ = self.attn1(ast_seq, clip_seq, clip_seq)   # (B, 1, 512)
        o2, _ = self.attn2(clip_seq, ast_seq, ast_seq)    # (B, 1, 512)

        fused = torch.cat([o1.squeeze(1), o2.squeeze(1)], dim=-1)  # (B, 1024)
        out = self.fusion(fused)                       # (B, 512)
        return F.normalize(out, p=2, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    B = 8
    clip = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    ast = F.normalize(torch.randn(B, config.AST_DIM), dim=-1)
    model = CrossModalDistillation()
    out = model(clip, ast)
    n_params = sum(p.numel() for p in model.parameters())
    assert out.shape == (B, config.EMB_DIM), out.shape
    assert torch.allclose(out.norm(dim=-1), torch.ones(B), atol=1e-5), "not unit norm"
    print(f"CrossModalDistillation ok, out {tuple(out.shape)}, params={n_params:,}")
