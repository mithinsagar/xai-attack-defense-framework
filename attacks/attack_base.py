"""
=====================================================================
XAI Attack and Defense Framework - Attack Base Class
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Abstract base class shared by all four explanation attacks. Each concrete
attack overrides `perturb` to return a modified input sample; the base
class handles the boilerplate of running the same attack across a list of
samples and computing explanation drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class AttackResult:
    """Container for the output of a single-sample attack."""
    original: np.ndarray
    perturbed: np.ndarray
    original_prediction: float | int
    perturbed_prediction: float | int
    original_prob: np.ndarray | None
    perturbed_prob: np.ndarray | None
    mean_drift: float
    max_drift: float
    top5_drift: float


class BaseAttack(ABC):
    """Base class for all explanation attacks."""

    name: str = "base_attack"
    epsilon: float = 0.05

    def __init__(self, feature_names: Sequence[str], epsilon: float | None = None):
        self.feature_names = list(feature_names)
        if epsilon is not None:
            self.epsilon = epsilon

    @abstractmethod
    def perturb(self, x: np.ndarray, model, explainer=None) -> np.ndarray:
        """Return a perturbed copy of the input sample `x`."""
