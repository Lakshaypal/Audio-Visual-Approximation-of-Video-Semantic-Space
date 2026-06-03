"""Model cost: parameter count, FLOPs and latency.

For each model it prints a row and appends it to results/efficiency.csv. FLOPs are
counted on CPU with fvcore so they don't depend on the device; latency is timed on
the actual device. The asymmetric model runs its no-teacher path here, the same as
it would at inference.
"""
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.asymmetric_distillation import AsymmetricDistillation
from models.baseline_regression import LatentSpaceRegression
from models.contrastive_alignment import ContrastiveAlignment
from models.cross_modal_distillation import CrossModalDistillation

MODELS = {
    "regression": LatentSpaceRegression,
    "contrastive": ContrastiveAlignment,
    "distillation": CrossModalDistillation,
    "asymmetric": AsymmetricDistillation,
}


def param_count(model) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def flops(model) -> float | None:
    """FLOPs via fvcore on CPU, or None if fvcore isn't installed.

    fvcore doesn't count every fused attention op, so for the attention models this
    is a lower bound. It's still fine for comparing the models against each other.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    model = model.to("cpu").eval()
    clip = torch.randn(1, config.EMB_DIM)
    ast = torch.randn(1, config.AST_DIM)
    fca = FlopCountAnalysis(model, (clip, ast))
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    fca.tracer_warnings("none")
    return float(fca.total())


def latency(model, device, n_warmup: int = 10, n_runs: int = 100) -> float:
    """Average milliseconds per single-sample forward pass on the device."""
    model = model.to(device).eval()
    clip = torch.randn(1, config.EMB_DIM, device=device)
    ast = torch.randn(1, config.AST_DIM, device=device)

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(clip, ast)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(clip, ast)
        sync()
        t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1000.0


def run(device: str = config.DEVICE) -> pd.DataFrame:
    config.ensure_dirs()
    rows = []
    for name, cls in MODELS.items():
        model = cls()
        total, trainable = param_count(model)
        fl = flops(model)                      # leaves the model on cpu
        lat = latency(model, device)           # moves the model to the device
        rows.append({
            "model": name,
            "total_params": total,
            "trainable_params": trainable,
            "flops": fl,
            "latency_ms": round(lat, 4),
            "device": device,
        })

    df = pd.DataFrame(rows)

    print(f"{'model':<14}{'total':>12}{'trainable':>12}{'FLOPs':>14}{'lat ms/smp':>12}")
    print("-" * 64)
    for r in rows:
        fl = "n/a" if r["flops"] is None else f"{r['flops']:,.0f}"
        print(f"{r['model']:<14}{r['total_params']:>12,}{r['trainable_params']:>12,}"
              f"{fl:>14}{r['latency_ms']:>12.4f}")

    csv_path = os.path.join(config.RESULTS_DIR, "efficiency.csv")
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)
    print(f"\n[efficiency] appended -> {csv_path}  (device={device})")
    return df


if __name__ == "__main__":
    run()
