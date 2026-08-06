"""
=====================================================================
XAI Attack and Defense Framework - Miscellaneous Helpers
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


def to_tensor(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a NumPy array to a torch tensor with the requested dtype."""
    return torch.tensor(x, dtype=dtype)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach a tensor from the graph and move to CPU as NumPy."""
    return x.detach().cpu().numpy()


def top_k_indices(values: Sequence[float], k: int, largest: bool = True) -> np.ndarray:
    """Return the indices of the top-k (or bottom-k) values of an array."""
    arr = np.asarray(values)
    if largest:
        return np.argsort(arr)[-k:][::-1]
    return np.argsort(arr)[:k]


def feature_importance_table(
    feature_names: Sequence[str], importance: Sequence[float]
) -> pd.DataFrame:
    """
    Build a `DataFrame(feature, importance)` sorted by descending importance.
    """
    table = pd.DataFrame(
        {"Feature": list(feature_names), "Importance": np.abs(importance)}
    )
    return table.sort_values("Importance", ascending=False).reset_index(drop=True)


def batch_iterator(x: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    """Yield contiguous batches of `batch_size` rows from `x`."""
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]
