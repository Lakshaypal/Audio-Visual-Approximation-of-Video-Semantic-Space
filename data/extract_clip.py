"""Extract one CLIP ViT-B/32 frame embedding per video (run on the Mac).

For each video_id in the S3D index it uses a pre-extracted JPG if there is one,
otherwise the middle frame of the mp4, and saves a dict {video_id: tensor(512)}
to config.CLIP_EMB. This is the step that produced clip_embeddings.pt. The
video_ids have to match the s3d_index.json ids exactly.

  python data/extract_clip.py --frames_dir data/frames
  python data/extract_clip.py --videos_dir data/videos
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _load_middle_frame(mp4_path):
    """Return the middle frame of an mp4 as a PIL image."""
    from PIL import Image
    try:
        import torchvision  # uses the pyav video backend if it's installed
        frames, _, _ = torchvision.io.read_video(mp4_path, pts_unit="sec")
        mid = frames[len(frames) // 2]                  # (H, W, C) uint8
        return Image.fromarray(mid.numpy())
    except Exception:
        import cv2
        cap = cv2.VideoCapture(mp4_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 2, 0))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"could not read middle frame from {mp4_path}")
        return Image.fromarray(frame[:, :, ::-1])       # cv2 is BGR, flip to RGB


def extract_clip(frames_dir=None, videos_dir=None, index_path=config.S3D_INDEX,
                 out_path=config.CLIP_EMB, batch_size=64) -> dict:
    import clip                                          # openai CLIP
    from PIL import Image

    device = config.DEVICE
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    with open(index_path) as f:
        ids = [r["video_id"] for r in json.load(f)]

    def resolve(vid):
        if frames_dir:
            jpg = os.path.join(frames_dir, f"{vid}.jpg")
            if os.path.exists(jpg):
                return Image.open(jpg).convert("RGB")
        if videos_dir:
            mp4 = os.path.join(videos_dir, f"{vid}.mp4")
            if os.path.exists(mp4):
                return _load_middle_frame(mp4)
        return None

    embeddings, skipped = {}, 0
    batch_ids, batch_imgs = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(x).float().cpu()  # (B, 512)
        for vid, f in zip(batch_ids, feats):
            embeddings[vid] = f
        batch_ids.clear()
        batch_imgs.clear()

    for vid in tqdm(ids, desc="CLIP"):
        try:
            img = resolve(vid)
            if img is None:
                skipped += 1
                continue
            batch_imgs.append(preprocess(img))
            batch_ids.append(vid)
            if len(batch_imgs) >= batch_size:
                flush()
        except Exception as e:                            # skip and keep going
            skipped += 1
            print(f"  [skip] {vid}: {e}")
    flush()

    torch.save(embeddings, out_path)
    print(f"extract_clip: {len(embeddings)} embeddings ({skipped} skipped) -> {out_path}")
    return embeddings


def build_argparser():
    p = argparse.ArgumentParser(description="CLIP ViT-B/32 frame extractor")
    p.add_argument("--frames_dir", default=None, help="dir of <video_id>.jpg")
    p.add_argument("--videos_dir", default=None, help="dir of <video_id>.mp4")
    p.add_argument("--index", default=config.S3D_INDEX)
    p.add_argument("--out", default=config.CLIP_EMB)
    p.add_argument("--batch_size", type=int, default=64)
    return p


if __name__ == "__main__":
    a = build_argparser().parse_args()
    if not a.frames_dir and not a.videos_dir:
        raise SystemExit("provide --frames_dir and/or --videos_dir")
    extract_clip(a.frames_dir, a.videos_dir, a.index, a.out, a.batch_size)
