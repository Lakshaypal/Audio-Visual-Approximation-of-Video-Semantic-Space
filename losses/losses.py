"""Training objectives for the four systems.

The regression baseline uses mse_cosine_loss, the contrastive model uses
infonce_loss, and both distillation models use crd_loss. Inputs that go into the
cosine / similarity terms are assumed to be L2-normalized already, so they are
not normalized again here.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def mse_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Half MSE plus half (1 - cosine). Used by the regression baseline."""
    cos = F.cosine_similarity(pred, target, dim=-1).mean()
    return 0.5 * F.mse_loss(pred, target) + 0.5 * (1.0 - cos)


def infonce_loss(
    av_emb: torch.Tensor,
    video_emb: torch.Tensor,
    temperature: float = config.TEMPERATURE,
) -> torch.Tensor:
    """Symmetric InfoNCE over the in-batch similarity matrix.

    The matched (i, i) pair is the positive and the other videos in the batch are
    the negatives, scored in both directions.
    """
    sim = av_emb @ video_emb.T / temperature        # (B, B)
    labels = torch.arange(sim.size(0), device=sim.device)
    return 0.5 * F.cross_entropy(sim, labels) + 0.5 * F.cross_entropy(sim.T, labels)


def crd_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    temperature: float = config.TEMPERATURE,
    rkd_weight: float = config.RKD_WEIGHT,
) -> torch.Tensor:
    """Contrastive distillation plus a relational (RKD) distance term.

    The contrastive part pulls each student embedding onto its own teacher. The
    RKD part matches the student and teacher pairwise-distance matrices, so the
    student keeps the same relative geometry as the teacher.
    """
    sim = student_emb @ teacher_emb.T / temperature  # (B, B)
    labels = torch.arange(sim.size(0), device=sim.device)
    contrastive = 0.5 * F.cross_entropy(sim, labels) + 0.5 * F.cross_entropy(sim.T, labels)

    sd = torch.cdist(student_emb, student_emb)       # (B, B) student distances
    td = torch.cdist(teacher_emb, teacher_emb)       # (B, B) teacher distances
    rkd = F.mse_loss(sd, td)

    return contrastive + rkd_weight * rkd


# which loss each --model name trains with (used by train/train.py)
LOSS_BY_MODEL = {
    "regression": "mse_cosine",
    "contrastive": "infonce",
    "distillation": "crd",
    "asymmetric": "crd",
    "asymmetric_gate": "crd",
}


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    B = 16
    pred = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)
    target = F.normalize(torch.randn(B, config.EMB_DIM), dim=-1)

    lm = mse_cosine_loss(pred, target)
    li = infonce_loss(pred, target)
    lc = crd_loss(pred, target)
    for name, val in (("mse_cosine", lm), ("infonce", li), ("crd", lc)):
        assert torch.isfinite(val) and val.ndim == 0, f"{name} not a finite scalar"
        print(f"  {name:11s} = {val.item():.4f}")

    # identical pred/target should drive mse_cosine to ~0
    same = mse_cosine_loss(target, target)
    assert same.item() < 1e-5, f"mse_cosine on identical inputs should be ~0, got {same.item()}"
    print(f"  mse_cosine(identical) = {same.item():.2e}")
    print("losses ok")
