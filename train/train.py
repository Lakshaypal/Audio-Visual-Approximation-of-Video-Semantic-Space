"""Training loop shared by all five models.

The --model flag picks both the network and the loss; everything else (optimizer,
schedule, evaluation) is the same. Only cached embeddings are read, never the
encoders. Each run writes a best checkpoint and a per-epoch CSV of the loss and
the retrieval metrics.

  python train/train.py --model regression
  python train/train.py --model contrastive
  python train/train.py --model distillation
  python train/train.py --model asymmetric
  python train/train.py --model asymmetric_gate
"""
import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.dataset import AudioVisualDataset
from eval.retrieval_eval import retrieval_eval
from losses.losses import crd_loss, infonce_loss, mse_cosine_loss
from models.asymmetric_distillation import AsymmetricDistillation
from models.asymmetric_gate import AsymmetricGateDistillation
from models.baseline_regression import LatentSpaceRegression
from models.contrastive_alignment import ContrastiveAlignment
from models.cross_modal_distillation import CrossModalDistillation
from utils.seed import set_seed

MODELS = {
    "regression": LatentSpaceRegression,
    "contrastive": ContrastiveAlignment,
    "distillation": CrossModalDistillation,
    "asymmetric": AsymmetricDistillation,
    "asymmetric_gate": AsymmetricGateDistillation,
}


def compute_loss(model_name, model, batch, device, temperature):
    """Run the forward pass and the loss that goes with the chosen model."""
    clip = batch["clip_emb"].to(device)
    ast = batch["ast_emb"].to(device)
    teacher = batch["video_emb"].to(device)     # S3D ground truth

    if model_name in ("asymmetric", "asymmetric_gate"):
        # these two pass the teacher into forward to compute the modality weights
        pred = model(clip, ast, teacher)
        model.update_class_alpha(batch["label"])
        return crd_loss(pred, teacher, temperature, config.RKD_WEIGHT)

    pred = model(clip, ast)
    if model_name == "regression":
        return mse_cosine_loss(pred, teacher)
    if model_name == "contrastive":
        return infonce_loss(pred, teacher, temperature)
    if model_name == "distillation":
        return crd_loss(pred, teacher, temperature, config.RKD_WEIGHT)
    raise ValueError(f"unknown model {model_name}")


def train(args):
    set_seed(config.SEED)
    config.ensure_dirs()
    device = config.DEVICE
    print(f"[train] model={args.model}  device={device}  epochs={args.epochs}  "
          f"batch={args.batch_size}  lr={args.lr}  temp={args.temperature}")

    train_ds = AudioVisualDataset("train")
    test_ds = AudioVisualDataset("test")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=config.PIN_MEMORY,
        drop_last=True,  # so the in-batch contrastive matrix stays full size
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=config.PIN_MEMORY,
    )

    model = MODELS[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {model.__class__.__name__}  trainable params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=config.WEIGHT_DECAY)
    # one cosine decay over the whole run, so step the scheduler once per epoch
    # (T_max is in epochs). stepping per batch would hit eta_min almost at once.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    rows = []
    best_r1 = -1.0
    best_path = os.path.join(args.save_dir, f"best_{args.model}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(args.model, model, batch, device, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            total += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total / max(n_batches, 1)

        # evaluate every few epochs and always on the last one, so the CSV and
        # the saved checkpoint always carry real retrieval numbers
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        row = {"epoch": epoch, "loss": avg_loss, "R@1": float("nan"),
               "R@5": float("nan"), "R@10": float("nan"), "MedR": float("nan"),
               "cosine": float("nan")}
        if do_eval:
            m = retrieval_eval(model, test_loader, device)
            row.update(m)
            lr_now = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"lr={lr_now:.2e}  R@1={m['R@1']:.2f}  R@5={m['R@5']:.2f}  "
                  f"R@10={m['R@10']:.2f}  MedR={m['MedR']:.1f}  cos={m['cosine']:.4f}")
            if m["R@1"] > best_r1:
                best_r1 = m["R@1"]
                torch.save(
                    {
                        "model": args.model,
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "R@1": m["R@1"],
                        "config": {k: getattr(config, k) for k in (
                            "EMB_DIM", "AST_DIM", "PROJ_DIM", "SEED", "LR",
                            "WEIGHT_DECAY", "TEMPERATURE", "RKD_WEIGHT",
                            "BATCH_SIZE", "EPOCHS", "GRAD_CLIP")},
                    },
                    best_path,
                )
        else:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}")
        rows.append(row)

    csv_path = os.path.join(config.RESULTS_DIR, f"training_log_{args.model}.csv")
    pd.DataFrame(rows, columns=["epoch", "loss", "R@1", "R@5", "R@10", "MedR",
                                "cosine"]).to_csv(csv_path, index=False)
    print(f"[train] done. best R@1={best_r1:.2f} -> {best_path}")
    print(f"[train] wrote {csv_path}")
    return csv_path


def build_argparser():
    p = argparse.ArgumentParser(description="train one of the AV-approximation models")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--save_dir", default=config.CKPT_DIR)
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--eval_every", type=int, default=5)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
