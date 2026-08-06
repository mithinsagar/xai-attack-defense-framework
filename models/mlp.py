"""
=====================================================================
XAI Attack and Defense Framework - MLP Base Model
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Reference Multi-Layer Perceptron used as the deep learning base model in
this project.

    Input (d)
        -> Linear(d, 256) + BN + ReLU + Dropout(0.3)
        -> Linear(256, 128) + BN + ReLU + Dropout(0.2)
        -> Linear(128, 64)  + ReLU
        -> Linear(64, 1)    + Sigmoid

The single-logit + Sigmoid + BCELoss configuration matches Phase 2 of the
original notebooks. `MLP2Class` provides the 2-logit + CrossEntropyLoss
variant used by the defense phase (where Integrated Gradients needs a
target class).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Base MLP: 256 -> 128 -> 64 -> 1 (Sigmoid)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLP2Class(nn.Module):
    """2-output logit variant for CrossEntropyLoss and targeted IG."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPSmall(nn.Module):
    """
    Compact MLP used for the small IDS dataset. Smaller capacity + no
    BatchNorm/Dropout to avoid overfitting on ~9.5K samples.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
