"""Figures: the 2x2 UMAP comparison and the Method 3 modality-weight bar chart.

umap_figure fits a single UMAP on the union of the four embedding sets (CLIP, AST
projected to 512, our model, and the S3D teacher), so all four panels share one
projection and can be compared directly. modality_weight_bar plots the per-class
mean audio weight from Method 3. If umap-learn isn't installed we fall back to PCA
so a figure is always produced.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")               # no display, just write files
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

PANELS = ["clip", "ast", "model", "s3d"]
PANEL_TITLES = {
    "clip": "(a) CLIP-only (512)",
    "ast": "(b) AST-only (proj→512)",
    "model": "(c) Our model",
    "s3d": "(d) S3D teacher",
}


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    else:
        x = torch.as_tensor(np.asarray(x))
    return F.normalize(x.float(), p=2, dim=-1).numpy()   # normalize before UMAP


def _project_ast_to_emb_dim(ast: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    # AST is 768-d; project it to 512 so all four panels share one UMAP. this is
    # just a fixed random projection for the plot, not the model's audio_proj.
    if ast.shape[1] == config.EMB_DIM:
        return ast
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(ast.shape[1], config.EMB_DIM, generator=g) / (ast.shape[1] ** 0.5)
    return (torch.from_numpy(ast).float() @ W).numpy()


def _reduce_2d(union: np.ndarray) -> np.ndarray:
    # one reducer fit on everything; UMAP if it's installed, otherwise PCA
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                            random_state=42)
        return reducer.fit_transform(union)
    except Exception as e:
        from sklearn.decomposition import PCA
        print(f"[visualize] UMAP unavailable ({type(e).__name__}); using PCA instead.")
        return PCA(n_components=2, random_state=42).fit_transform(union)


def umap_figure(embeddings_dict: dict, labels, out_path: str) -> str:
    config.ensure_dirs()
    labels = np.asarray(labels)

    clip = _to_np(embeddings_dict["clip"])
    ast = _project_ast_to_emb_dim(_to_np(embeddings_dict["ast"]))
    model = _to_np(embeddings_dict["model"])
    s3d = _to_np(embeddings_dict["s3d"])
    panels = {"clip": clip, "ast": ast, "model": model, "s3d": s3d}

    n = clip.shape[0]
    union = np.concatenate([panels[k] for k in PANELS], axis=0)   # (4N, 512)
    coords = _reduce_2d(union)                                    # (4N, 2)
    per_panel = coords.reshape(len(PANELS), n, 2)                 # (4, N, 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, key in zip(axes.flat, PANELS):
        c = per_panel[PANELS.index(key)]
        ax.scatter(c[:, 0], c[:, 1], c=labels % 20, cmap="tab20", s=12, alpha=0.8)
        ax.set_title(PANEL_TITLES[key], fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("UMAP of embedding spaces (shared projection)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # save the 2-D coordinates so the figure can be redrawn without refitting
    coords_path = os.path.splitext(out_path)[0] + "_coords.npy"
    np.save(coords_path, per_panel)              # (4, N, 2)
    print(f"[visualize] saved {out_path}  (+ coords {os.path.basename(coords_path)})")
    return out_path


def modality_weight_bar(class_alpha_dict: dict, out_path: str) -> str:
    config.ensure_dirs()
    classes = sorted(class_alpha_dict.keys())
    alphas = [class_alpha_dict[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.3), 4))
    ax.bar(range(len(classes)), alphas, color="tab:blue")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="equal weight (0.5)")
    ax.set_xlabel("class index")
    ax.set_ylabel("mean audio weight α")
    ax.set_title("Per-class modality confidence (Method 3)")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved {out_path}")
    return out_path


def _demo() -> None:
    # build both figures from the trained Method 3 checkpoint on the real test set
    from torch.utils.data import DataLoader

    from data.dataset import AudioVisualDataset
    from models.asymmetric_distillation import AsymmetricDistillation

    ds = AudioVisualDataset("test")
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = AsymmetricDistillation().to(config.DEVICE)
    ckpt_path = os.path.join(config.CKPT_DIR, "best_asymmetric.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        print(f"[visualize] loaded trained model {ckpt_path} "
              f"(epoch {ckpt.get('epoch', '?')}, R@1={ckpt.get('R@1', '?')})")
    else:
        print(f"[visualize] warning: {ckpt_path} not found, using untrained weights")
    model.eval()

    clip, ast, s3d, mdl, labels = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            c = batch["clip_emb"].to(config.DEVICE)
            a = batch["ast_emb"].to(config.DEVICE)
            t = batch["video_emb"].to(config.DEVICE)
            out = model(c, a)                    # no-teacher path, for the UMAP panel
            model(c, a, t)                       # teacher path, just to log alpha
            model.update_class_alpha(batch["label"])
            clip.append(c.cpu())
            ast.append(a.cpu())
            s3d.append(t.cpu())
            mdl.append(out.cpu())
            labels.append(batch["label"])
    emb = {
        "clip": torch.cat(clip), "ast": torch.cat(ast),
        "model": torch.cat(mdl), "s3d": torch.cat(s3d),
    }
    labels = torch.cat(labels).numpy()

    avals = list(model.class_alpha.values())
    print(f"[visualize] per-class audio weight: "
          f"min={min(avals):.3f} max={max(avals):.3f} "
          f"mean={sum(avals) / len(avals):.3f} (n_classes={len(avals)})")
    umap_figure(emb, labels, os.path.join(config.RESULTS_DIR, "umap_real.png"))
    modality_weight_bar(model.class_alpha,
                        os.path.join(config.RESULTS_DIR, "modality_weights_real.png"))
    print("visualize done, figures written to results/")


if __name__ == "__main__":
    _demo()
