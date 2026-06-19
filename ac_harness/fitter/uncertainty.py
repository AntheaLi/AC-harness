# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Bootstrap uncertainty helpers for the fitters (§12).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_std(
    fit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Returns per-coefficient standard error via simple bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return np.zeros(X.shape[1])
    coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        coefs.append(fit_fn(X[idx], y[idx]))
    arr = np.stack(coefs)
    return arr.std(axis=0)
