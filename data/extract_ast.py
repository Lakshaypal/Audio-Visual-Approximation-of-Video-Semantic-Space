"""Extract one AST audio embedding per video (run on the Mac).

For each video_id in the S3D index it loads the mono WAV at 16 kHz, runs the
AudioSet AST model, and saves a dict {video_id: tensor(768)} to config.AST_EMB.
The feature extractor builds the log-mel spectrogram the model expects. This is
the step that produced ast_embeddings.pt. The video_ids have to match
s3d_index.json exactly.

  python data/extract_ast.py --audio_dir data/audio
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP = 160
T_FRAMES = 1024


def extract_ast(audio_dir, index_path=config.S3D_INDEX, out_path=config.AST_EMB,
                batch_size=32) -> dict:
    import librosa
    from transformers import ASTFeatureExtractor, ASTModel

    device = config.DEVICE
    extractor = ASTFeatureExtractor.from_pretrained(
        AST_MODEL, num_mel_bins=N_MELS, max_length=T_FRAMES, sampling_rate=SAMPLE_RATE)
    model = ASTModel.from_pretrained(AST_MODEL).to(device).eval()

    with open(index_path) as f:
        ids = [r["video_id"] for r in json.load(f)]

    embeddings, skipped = {}, 0
    batch_ids, batch_wavs = [], []

    def flush():
        if not batch_wavs:
            return
        inputs = extractor(batch_wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            feats = out.pooler_output.float().cpu()       # (B, 768)
        for vid, f in zip(batch_ids, feats):
            embeddings[vid] = f
        batch_ids.clear()
        batch_wavs.clear()

    for vid in tqdm(ids, desc="AST"):
        wav_path = os.path.join(audio_dir, f"{vid}.wav")
        if not os.path.exists(wav_path):
            skipped += 1
            continue
        try:
            wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            batch_wavs.append(wav)
            batch_ids.append(vid)
            if len(batch_wavs) >= batch_size:
                flush()
        except Exception as e:                            # skip and keep going
            skipped += 1
            print(f"  [skip] {vid}: {e}")
    flush()

    torch.save(embeddings, out_path)
    print(f"extract_ast: {len(embeddings)} embeddings ({skipped} skipped) -> {out_path}")
    return embeddings


def build_argparser():
    p = argparse.ArgumentParser(description="AST audio extractor")
    p.add_argument("--audio_dir", required=True, help="dir of <video_id>.wav")
    p.add_argument("--index", default=config.S3D_INDEX)
    p.add_argument("--out", default=config.AST_EMB)
    p.add_argument("--batch_size", type=int, default=32)
    return p


if __name__ == "__main__":
    a = build_argparser().parse_args()
    extract_ast(a.audio_dir, a.index, a.out, a.batch_size)
