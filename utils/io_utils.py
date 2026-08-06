"""
=====================================================================
XAI Attack and Defense Framework - I/O Utilities
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Thin wrappers around pickle / numpy / torch save-and-load, plus JSON
helpers. Centralising these avoids scattering `with open(...)` blocks
across the codebase.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------
# Pickle
# ---------------------------------------------------------------------
def save_pickle(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | os.PathLike) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------
# Numpy
# ---------------------------------------------------------------------
def save_numpy(arr: np.ndarray, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_numpy(path: str | os.PathLike, allow_pickle: bool = False) -> np.ndarray:
    return np.load(path, allow_pickle=allow_pickle)


# ---------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------
def save_torch_model(model: torch.nn.Module, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_torch_model(
    model: torch.nn.Module,
    path: str | os.PathLike,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------
def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
