"""Small shared runtime helpers."""

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set the random seeds and deterministic CuDNN options used in experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python/NumPy augmentation RNGs from a DataLoader worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
