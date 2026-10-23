"""Randomness and Reproducibility Utilities for Q-Nav.

Ensures deterministic execution across classical and quantum simulations.
"""

import os
import random
from typing import Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set global seeds across Python standard library, NumPy, and environment.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create and return an isolated NumPy random number generator.

    Args:
        seed: Optional seed. If None, default 42 is used.

    Returns:
        np.random.Generator instance.
    """
    if seed is None:
        seed = 42
    return np.random.default_rng(seed)
