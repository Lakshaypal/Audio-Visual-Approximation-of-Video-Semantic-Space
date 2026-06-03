"""Consolidate and sanity-check the three caches.

Reads the S3D index, the CLIP cache and the AST cache, keeps the videos present
in all three, and writes one dataset.pt mapping each video_id to its three raw
embeddings plus label and split. The Dataset normalizes at read time, so the
embeddings are stored un-normalized here. This is mostly a coverage check;
dataset.py can read the three caches directly.

  python data/merge_embeddings.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DATASET_PT = os.path.join(config.DATA_DIR, "dataset.pt")


def merge_embeddings(index_path=config.S3D_INDEX, clip_path=config.CLIP_EMB,
                     ast_path=config.AST_EMB, s3d_dir=config.S3D_EMB_DIR,
                     out_path=DATASET_PT) -> dict:
    with open(index_path) as f:
        index = json.load(f)
    clip_db = torch.load(clip_path)
    ast_db = torch.load(ast_path)

    merged = {}
    dropped = {"clip": 0, "ast": 0, "s3d": 0}
    per_split = {}
    for r in index:
        vid = r["video_id"]
        if vid not in clip_db:
            dropped["clip"] += 1
            continue
        if vid not in ast_db:
            dropped["ast"] += 1
            continue
        s3d_path = r.get("embedding_path") or os.path.join(s3d_dir, f"{vid}.pt")
        if not os.path.exists(s3d_path):
            dropped["s3d"] += 1
            continue
        video_emb = torch.load(s3d_path)["emb"]
        merged[vid] = {
            "clip": clip_db[vid],
            "ast": ast_db[vid],
            "video": video_emb,
            "label": r["label"],
            "split": r["split"],
        }
        per_split[r["split"]] = per_split.get(r["split"], 0) + 1

    torch.save(merged, out_path)
    print("merge_embeddings: consolidated cache written")
    print(f"  index rows      : {len(index)}")
    print(f"  merged samples  : {len(merged)}  by split: {per_split}")
    print(f"  dropped         : clip={dropped['clip']} ast={dropped['ast']} "
          f"s3d={dropped['s3d']}")
    print(f"  -> {out_path}")
    return merged


if __name__ == "__main__":
    m = merge_embeddings()
    if m:
        v = next(iter(m.values()))
        assert v["clip"].shape[-1] == config.EMB_DIM
        assert v["ast"].shape[-1] == config.AST_DIM
        assert v["video"].shape[-1] == config.EMB_DIM
        print("  shapes ok: clip(512), ast(768), video(512)")
