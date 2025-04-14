"""
Utilities for seeding random number generators.
"""

import random
import numpy as np
import torch


def seed_worker(worker_id, seed=42):
    """
    Function to set seed for DataLoader workers to ensure reproducibility.

    Args:
        worker_id: The ID of the worker
        seed: The base seed value to use
    """
    # Different seed for each worker based on worker_id
    worker_seed = seed + worker_id
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seed_all(seed=42):
    """
    Set seed for all random number generators to ensure reproducibility.

    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set MPS seed if available
    if (
        hasattr(torch.mps, "manual_seed")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.manual_seed(seed)

    # Set environment variables for additional sources of randomness
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
