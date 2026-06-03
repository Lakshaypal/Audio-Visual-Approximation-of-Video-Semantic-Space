"""Seed Python, NumPy and torch so a run is reproducible."""
import os
import random
import sys

import numpy as np
import torch

# repo root on the path so `import config` works when this file is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed all the RNGs the project uses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed()
    a = torch.randn(3)
    set_seed()
    b = torch.randn(3)
    assert torch.equal(a, b), "set_seed is not deterministic!"
    print(f"set_seed ok, reproduced: {a.tolist()}")
