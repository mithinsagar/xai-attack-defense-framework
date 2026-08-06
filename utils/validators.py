"""
=====================================================================
XAI Attack and Defense Framework - Input Validators
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def require_2d(x: np.ndarray, name: str = "array") -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {x.shape}")


def require_1d(x: np.ndarray, name: str = "array") -> None:
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {x.shape}")


def require_same_length(a, b, name_a: str = "a", name_b: str = "b") -> None:
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} and {name_b} must have the same length "
            f"({len(a)} vs {len(b)})"
        )


def require_existing_file(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")
    return p


def require_probability(x: float, name: str = "value") -> None:
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {x}")
