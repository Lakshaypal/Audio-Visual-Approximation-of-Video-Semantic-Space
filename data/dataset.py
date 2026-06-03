"""Dataset of cached audio-visual embeddings.

For each video it returns the CLIP frame embedding, the AST audio embedding and
the S3D teacher embedding (all L2-normalized), plus the label and the video id.
Only cached tensors are read here; the encoders are never run.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _l2(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


class AudioVisualDataset(Dataset):
    def __init__(
        self,
        split: str,
        index_path: str = config.S3D_INDEX,
        clip_path: str = config.CLIP_EMB,
        ast_path: str = config.AST_EMB,
        s3d_dir: str = config.S3D_EMB_DIR,
    ):
        self.split = split

        with open(index_path) as f:
            index = json.load(f)

        # build the label -> id map from the whole index so train and test agree
        all_labels = sorted({row["label"] for row in index})
        self.label2idx = {lab: i for i, lab in enumerate(all_labels)}
        self.num_classes = len(all_labels)

        clip_db = torch.load(clip_path)
        ast_db = torch.load(ast_path)

        rows = [r for r in index if r["split"] == split]

        # keep only videos that have all three embeddings; drop the rest here
        # (with a count) instead of failing later inside __getitem__
        self.samples = []          # (video_id, label_idx) pairs
        self.clip = {}
        self.ast = {}
        self.video = {}
        dropped = {"missing_clip": 0, "missing_ast": 0, "missing_s3d": 0, "bad_s3d": 0}

        for r in rows:
            vid = r["video_id"]
            if vid not in clip_db:
                dropped["missing_clip"] += 1
                continue
            if vid not in ast_db:
                dropped["missing_ast"] += 1
                continue
            # load the teacher from our cache folder by id; the index's
            # embedding_path points at the original extraction location, not here
            s3d_path = os.path.join(s3d_dir, f"{vid}.pt")
            if not os.path.exists(s3d_path):
                dropped["missing_s3d"] += 1
                continue
            try:
                payload = torch.load(s3d_path)
                emb = payload["emb"]
            except (KeyError, RuntimeError, EOFError):
                dropped["bad_s3d"] += 1
                continue

            # normalize once now so __getitem__ stays cheap
            self.clip[vid] = _l2(clip_db[vid])
            self.ast[vid] = _l2(ast_db[vid])
            self.video[vid] = _l2(emb)
            self.samples.append((vid, self.label2idx[r["label"]]))

        n_dropped = sum(dropped.values())
        print(
            f"[AudioVisualDataset:{split}] kept {len(self.samples)}/{len(rows)} "
            f"samples; dropped {n_dropped} "
            f"(clip={dropped['missing_clip']}, ast={dropped['missing_ast']}, "
            f"s3d_missing={dropped['missing_s3d']}, s3d_bad={dropped['bad_s3d']}); "
            f"num_classes={self.num_classes}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        vid, label = self.samples[i]
        return {
            "clip_emb": self.clip[vid],     # (512)
            "ast_emb": self.ast[vid],       # (768)
            "video_emb": self.video[vid],   # (512) teacher
            "label": label,
            "video_id": vid,
        }


if __name__ == "__main__":
    for split in ("train", "test"):
        ds = AudioVisualDataset(split)
        s = ds[0]
        print(f"  [{split}] len={len(ds)}  sample keys={list(s.keys())}")
        print(
            f"    clip_emb  {tuple(s['clip_emb'].shape)}  |norm|={s['clip_emb'].norm():.4f}\n"
            f"    ast_emb   {tuple(s['ast_emb'].shape)}  |norm|={s['ast_emb'].norm():.4f}\n"
            f"    video_emb {tuple(s['video_emb'].shape)}  |norm|={s['video_emb'].norm():.4f}\n"
            f"    label={s['label']}  video_id={s['video_id']}"
        )
        for k in ("clip_emb", "ast_emb", "video_emb"):
            assert abs(s[k].norm().item() - 1.0) < 1e-5, f"{k} not unit norm"
    print("dataset ok, embeddings are unit norm")
