"""Baseline 2: ImageBind zero-shot (the ceiling, optional).

No training. Encodes the same frame and audio with ImageBind, fuses them in
ImageBind's shared space, and reports retrieval there. It is run separately and
lives in ImageBind's own space, not S3D's, so it is a reference point rather than
a like-for-like comparison with the trained models.

Needs the ImageBind repo and checkpoint plus the frames/audio on disk. The heavy
imports are inside the functions so the module still imports without ImageBind.

  python eval/imagebind_eval.py --frames_dir data/frames --audio_dir data/audio
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from eval.retrieval_eval import retrieval_metrics


def _load_imagebind(device):
    from imagebind.models import imagebind_model
    model = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()
    return model


def imagebind_eval(frames_dir, audio_dir, index_path=config.S3D_INDEX) -> dict:
    from imagebind.models.imagebind_model import ModalityType
    from imagebind import data as ib_data

    device = config.DEVICE
    model = _load_imagebind(device)

    with open(index_path) as f:
        rows = [r for r in json.load(f) if r["split"] == "test"]

    vision_list, audio_list, av_list = [], [], []
    for r in tqdm(rows, desc="ImageBind"):
        vid = r["video_id"]
        jpg = os.path.join(frames_dir, f"{vid}.jpg")
        wav = os.path.join(audio_dir, f"{vid}.wav")
        if not (os.path.exists(jpg) and os.path.exists(wav)):
            continue
        inputs = {
            ModalityType.VISION: ib_data.load_and_transform_vision_data([jpg], device),
            ModalityType.AUDIO: ib_data.load_and_transform_audio_data([wav], device),
        }
        with torch.no_grad():
            emb = model(inputs)
        v = F.normalize(emb[ModalityType.VISION].cpu(), dim=-1)
        a = F.normalize(emb[ModalityType.AUDIO].cpu(), dim=-1)
        vision_list.append(v)
        audio_list.append(a)
        av_list.append(F.normalize(0.5 * (v + a), dim=-1))   # fused audio-visual query

    vision = torch.cat(vision_list)
    audio = torch.cat(audio_list)
    av = torch.cat(av_list)

    # both directions, scored inside ImageBind's shared space
    av_to_vision = retrieval_metrics(av, vision)
    audio_to_vision = retrieval_metrics(audio, vision)

    print("ImageBind zero-shot (Baseline 2)")
    print(f"  N test           : {av.shape[0]}")
    print(f"  AV  -> vision     : {av_to_vision}")
    print(f"  audio -> vision   : {audio_to_vision}")

    config.ensure_dirs()
    out_csv = os.path.join(config.RESULTS_DIR, "imagebind_zeroshot.csv")
    import pandas as pd
    pd.DataFrame([
        {"query": "av", **av_to_vision},
        {"query": "audio", **audio_to_vision},
    ]).to_csv(out_csv, index=False)
    print(f"  -> {out_csv}")
    return {"av_to_vision": av_to_vision, "audio_to_vision": audio_to_vision}


def build_argparser():
    p = argparse.ArgumentParser(description="ImageBind zero-shot baseline (ceiling)")
    p.add_argument("--frames_dir", required=True, help="dir of <video_id>.jpg")
    p.add_argument("--audio_dir", required=True, help="dir of <video_id>.wav")
    p.add_argument("--index", default=config.S3D_INDEX)
    return p


if __name__ == "__main__":
    a = build_argparser().parse_args()
    imagebind_eval(a.frames_dir, a.audio_dir, a.index)
