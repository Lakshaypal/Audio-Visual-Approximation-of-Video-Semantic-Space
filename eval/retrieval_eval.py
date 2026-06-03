"""Retrieval metrics plus a rigged sanity test.

For a model and the test loader, predict one audio-visual embedding per video and
rank it against all the S3D teacher embeddings. R@K is the percentage whose
correct video lands in the top K, MedR is the median rank (1-indexed) and cosine
is the mean similarity of the matched pairs.

The rigged test feeds identical predicted and ground-truth embeddings, which has
to give R@1 = 100% and MedR = 1. If that ever fails the metric code is wrong.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def retrieval_metrics(av_emb: torch.Tensor, video_emb: torch.Tensor) -> dict:
    """Compute the metrics from two (N, D) banks of embeddings."""
    av = F.normalize(av_emb.float(), p=2, dim=-1)
    video = F.normalize(video_emb.float(), p=2, dim=-1)

    sim = av @ video.T                          # (N, N) cosine similarities
    correct = sim.diag().view(-1, 1)            # similarity of each true match
    # rank = how many candidates score strictly higher than the true match
    ranks = (sim > correct).sum(dim=1)
    ranks_np = ranks.cpu().numpy()

    return {
        "R@1": float((ranks < 1).float().mean().item() * 100.0),
        "R@5": float((ranks < 5).float().mean().item() * 100.0),
        "R@10": float((ranks < 10).float().mean().item() * 100.0),
        "MedR": float(np.median(ranks_np) + 1.0),
        "cosine": float(sim.diag().mean().item()),
    }


@torch.no_grad()
def retrieval_eval(model, test_loader, device) -> dict:
    """Run the model over the test set and return the metrics."""
    model.eval()
    av_list, video_list = [], []
    for batch in test_loader:
        clip = batch["clip_emb"].to(device)
        ast = batch["ast_emb"].to(device)
        av = model(clip, ast)                   # (B, 512), already normalized
        av_list.append(av.detach().cpu())
        video_list.append(batch["video_emb"])   # ground truth, already normalized

    av_emb = torch.cat(av_list, dim=0)
    video_emb = torch.cat(video_list, dim=0)
    return retrieval_metrics(av_emb, video_emb)


def _run_rigged_test() -> None:
    # identical predicted and ground-truth embeddings -> perfect retrieval
    torch.manual_seed(config.SEED)
    N, D = 50, config.EMB_DIM
    emb = F.normalize(torch.randn(N, D), p=2, dim=-1)
    m = retrieval_metrics(emb, emb)
    print(f"  rigged  R@1={m['R@1']:.1f}  R@5={m['R@5']:.1f}  "
          f"R@10={m['R@10']:.1f}  MedR={m['MedR']:.1f}  cosine={m['cosine']:.4f}")
    assert m["R@1"] == 100.0, f"rigged fail: R@1={m['R@1']} (expected 100.0)"
    assert m["MedR"] == 1.0, f"rigged fail: MedR={m['MedR']} (expected 1.0)"
    assert m["cosine"] > 0.999, f"rigged fail: cosine={m['cosine']} (expected ~1.0)"
    print("  RIGGED TEST PASSED. identical embeddings give R@1=100%, MedR=1")


def _run_end_to_end() -> None:
    # run the whole model -> loader -> metrics path once on the real test set
    from torch.utils.data import DataLoader

    from data.dataset import AudioVisualDataset
    from models.baseline_regression import LatentSpaceRegression

    ds = AudioVisualDataset("test")
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=config.PIN_MEMORY)
    model = LatentSpaceRegression().to(config.DEVICE)
    m = retrieval_eval(model, loader, config.DEVICE)
    print(f"  end-to-end (untrained model, N={len(ds)}): "
          f"R@1={m['R@1']:.2f}  R@5={m['R@5']:.2f}  R@10={m['R@10']:.2f}  "
          f"MedR={m['MedR']:.1f}  cosine={m['cosine']:.4f}")
    assert set(m.keys()) == {"R@1", "R@5", "R@10", "MedR", "cosine"}
    print("  end-to-end ok, metrics dict has all keys")


if __name__ == "__main__":
    _run_rigged_test()
    _run_end_to_end()
