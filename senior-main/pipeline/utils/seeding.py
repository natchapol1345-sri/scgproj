"""Seed all RNG sources for reproducibility."""
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA), and Open3D."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import open3d as o3d
        o3d.utility.random.seed(seed)
    except Exception:
        pass
