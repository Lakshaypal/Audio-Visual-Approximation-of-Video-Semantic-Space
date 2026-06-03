"""Method 3 variant: hard audio-reliability gate (ablation).

Same model and fusion as AsymmetricDistillation, but the per-sample weight comes
from a sharp sigmoid gate on the audio score instead of a softmax over both
modalities:

  audio_score = cos(audio_proj(ast), teacher)
  gate        = sigmoid((audio_score - GATE_BIAS) * GATE_SHARPNESS)
  alpha, beta = gate, 1 - gate

So audio is trusted only when it is clearly close to the teacher, with a hard
cutoff rather than a smooth trade-off. Only the weight computation changes; the
fusion and everything else are inherited. Run with `--model asymmetric_gate`.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.asymmetric_distillation import AsymmetricDistillation


class AsymmetricGateDistillation(AsymmetricDistillation):
    def _confidence_weights(self, clip_emb, ast_proj, teacher_emb):
        B = clip_emb.size(0)
        if teacher_emb is None:
            # no teacher at eval time, so leave the gate neutral (equal weights)
            alpha = torch.full((B, 1), 0.5, device=clip_emb.device, dtype=clip_emb.dtype)
            return alpha, 1.0 - alpha
        # the sharp sigmoid is basically an on/off switch around GATE_BIAS. only
        # the audio score is needed here; the image side is just 1 - gate.
        audio_score = F.cosine_similarity(ast_proj, teacher_emb, dim=-1)  # (B,)
        audio_gate = torch.sigmoid((audio_score - config.GATE_BIAS) * config.GATE_SHARPNESS)
        alpha = audio_gate.unsqueeze(1)            # (B, 1) audio weight
        return alpha, 1.0 - alpha                  # beta = 1 - gate


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    B = 8
    clip = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    ast = F.normalize(torch.randn(B, config.AST_DIM), dim=-1)
    teacher = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    labels = torch.randint(0, 32, (B,))

    model = AsymmetricGateDistillation()
    n_params = sum(p.numel() for p in model.parameters())

    out_t = model(clip, ast, teacher)              # train-time path + alpha logging
    model.update_class_alpha(labels)
    assert out_t.shape == (B, config.EMB_DIM)
    assert torch.allclose(out_t.norm(dim=-1), torch.ones(B), atol=1e-5), "not unit norm"

    model(clip, ast)                               # eval path (no teacher) -> neutral gate
    assert torch.allclose(model._last_alpha, torch.full((B,), 0.5)), "eval gate should be 0.5"

    # a different teacher should move the gate and so the output
    teacher2 = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    assert not torch.allclose(out_t, model(clip, ast, teacher2), atol=1e-6), "gate has no effect"

    print(f"AsymmetricGateDistillation ok, params={n_params:,} "
          f"(bias={config.GATE_BIAS}, sharpness={config.GATE_SHARPNESS})")
    print(f"  class_alpha (sampled) = { {k: round(v, 3) for k, v in model.class_alpha.items()} }")
