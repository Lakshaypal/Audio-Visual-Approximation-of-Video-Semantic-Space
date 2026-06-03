"""Project configuration: paths, dimensions and training hyperparameters.

Everything is read from here so the same values are not duplicated across files.
Paths are built from this file's location so the scripts work from any directory.
"""
import os

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_device() -> str:
    # use the Apple Silicon GPU if it's there, otherwise fall back to CPU
    return "mps" if torch.backends.mps.is_available() else "cpu"


# core dims and seed
DEVICE = _resolve_device()
EMB_DIM = 512                       # CLIP and S3D embedding size
AST_DIM = 768                       # AST embedding size
SEED = 42

# cache and output locations
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
S3D_INDEX = os.path.join(DATA_DIR, "s3d_index.json")
CLIP_EMB = os.path.join(DATA_DIR, "clip_embeddings.pt")
AST_EMB = os.path.join(DATA_DIR, "ast_embeddings.pt")
S3D_EMB_DIR = os.path.join(DATA_DIR, "s3d_embeddings")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# training hyperparameters
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07                  # temperature for the InfoNCE / CRD softmax
RKD_WEIGHT = 0.1                    # weight on the relational term in CRD
PROJ_DIM = 256                      # stored in checkpoints; models project to EMB_DIM
GRAD_CLIP = 1.0

# audio-gate variant (--model asymmetric_gate)
GATE_BIAS = 0.3                     # cosine threshold above which we trust audio
GATE_SHARPNESS = 10                 # how sharp the sigmoid gate is around the bias

# dataloader (pin_memory has to stay off on MPS)
NUM_WORKERS = 2
PIN_MEMORY = False


def ensure_dirs() -> None:
    """Create the cache, checkpoint and results folders if they don't exist yet."""
    for d in (DATA_DIR, S3D_EMB_DIR, CKPT_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("resolved settings:")
    print(f"  PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"  DEVICE       = {DEVICE}")
    print(f"  EMB_DIM      = {EMB_DIM}  AST_DIM = {AST_DIM}  PROJ_DIM = {PROJ_DIM}")
    print(f"  DATA_DIR     = {DATA_DIR}")
    print(f"  CKPT_DIR     = {CKPT_DIR}")
    print(f"  RESULTS_DIR  = {RESULTS_DIR}")
