"""Method 3: asymmetric modality weighting.

Extends Method 2 with a per-sample weight on each modality. With length-1
sequences the attention output does not depend on the query (a softmax over a
single key is always 1), so weighting the query would do nothing. Instead we
weight the two attention OUTPUTS before fusing them:

  alpha, beta = softmax([cos(audio, teacher), cos(image, teacher)] / CONF_TEMP)
  fused = beta * image_out + alpha * audio_out
  out   = normalize(proj(fused))

The teacher is only available during training, so at eval we fall back to equal
weights (alpha = beta = 0.5). Per-class mean alpha is logged in self.class_alpha
for the bar chart. Method 2 is left untouched.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.cross_modal_distillation import CrossModalDistillation

CONF_TEMP = 0.5      # temperature on the [audio, image] confidence logits


class AsymmetricDistillation(CrossModalDistillation):
    def __init__(self):
        super().__init__()
        # we fuse a weighted sum (512) rather than a concat (1024), so the
        # projection here is 512->512 (this replaces Method 2's 1024->512 fusion)
        self.fusion = nn.Linear(config.EMB_DIM, config.EMB_DIM)

        self.class_alpha = {}           # label_idx -> running mean audio weight
        self._alpha_accum = {}          # label_idx -> (mean, count)
        self._last_alpha = None         # (B,) audio weight from the last forward

    def _confidence_weights(self, clip_emb, ast_proj, teacher_emb):
        """Return (alpha, beta) of shape (B, 1): the audio/image weight per sample."""
        B = clip_emb.size(0)
        if teacher_emb is None:
            # no teacher at eval time, so weight both modalities equally
            alpha = torch.full((B, 1), 0.5, device=clip_emb.device, dtype=clip_emb.dtype)
            return alpha, 1.0 - alpha
        audio_score = F.cosine_similarity(ast_proj, teacher_emb, dim=-1)   # (B,)
        image_score = F.cosine_similarity(clip_emb, teacher_emb, dim=-1)   # (B,)
        logits = torch.stack([audio_score, image_score], dim=1) / CONF_TEMP
        w = F.softmax(logits, dim=1)               # (B, 2)
        return w[:, 0:1], w[:, 1:2]                 # alpha (audio), beta (image)

    def forward(self, clip_emb, ast_emb, teacher_emb=None) -> torch.Tensor:
        ast_proj = self.audio_proj(ast_emb)            # (B, 512)
        clip_seq = clip_emb.unsqueeze(1)               # (B, 1, 512)
        ast_seq = ast_proj.unsqueeze(1)                # (B, 1, 512)

        # same two attentions as Method 2: one image-derived, one audio-derived
        o_image, _ = self.attn1(ast_seq, clip_seq, clip_seq)   # value = clip
        o_audio, _ = self.attn2(clip_seq, ast_seq, ast_seq)    # value = ast_proj
        o_image = o_image.squeeze(1)                   # (B, 512)
        o_audio = o_audio.squeeze(1)                   # (B, 512)

        alpha, beta = self._confidence_weights(clip_emb, ast_proj, teacher_emb)
        self._last_alpha = alpha.detach().squeeze(1)   # (B,) kept for logging

        fused = beta * o_image + alpha * o_audio       # (B, 512) weighted sum
        out = self.fusion(fused)                       # 512 -> 512
        return F.normalize(out, p=2, dim=-1)

    @torch.no_grad()
    def update_class_alpha(self, labels: torch.Tensor) -> None:
        """Add the last forward's per-sample alpha into the per-class running mean."""
        if self._last_alpha is None:
            return
        for lab, a in zip(labels.tolist(), self._last_alpha.tolist()):
            mean, n = self._alpha_accum.get(lab, (0.0, 0))
            n += 1
            mean += (a - mean) / n
            self._alpha_accum[lab] = (mean, n)
            self.class_alpha[lab] = mean


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    B = 8
    clip = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    ast = F.normalize(torch.randn(B, config.AST_DIM), dim=-1)
    teacher = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    labels = torch.randint(0, 32, (B,))

    model = AsymmetricDistillation()
    n_params = sum(p.numel() for p in model.parameters())

    out_t = model(clip, ast, teacher)              # train-time path + alpha logging
    model.update_class_alpha(labels)
    assert out_t.shape == (B, config.EMB_DIM)
    assert torch.allclose(out_t.norm(dim=-1), torch.ones(B), atol=1e-5), "not unit norm"

    out_e = model(clip, ast)                       # eval path (no teacher) -> equal weights
    assert out_e.shape == (B, config.EMB_DIM)
    assert torch.allclose(out_e.norm(dim=-1), torch.ones(B), atol=1e-5), "not unit norm"
    assert torch.allclose(model._last_alpha, torch.full((B,), 0.5)), "eval alpha should be 0.5"

    # a different teacher should change the weights and so the output
    teacher2 = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    out_t2 = model(clip, ast, teacher2)
    assert not torch.allclose(out_t, out_t2, atol=1e-6), "weighting has no effect"
    delta = (out_t - out_t2).abs().max().item()

    print(f"AsymmetricDistillation ok, params={n_params:,}, max delta on new teacher={delta:.4f}")
    print(f"  class_alpha (sampled) = { {k: round(v, 3) for k, v in model.class_alpha.items()} }")
