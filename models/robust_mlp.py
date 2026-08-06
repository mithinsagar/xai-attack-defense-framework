"""
=====================================================================
XAI Attack and Defense Framework - Robust MLP Architectures
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Two MLP architectures used by the defense phase (Phase 6):

    RobustMLP     : Simple 128 -> 64 -> 2 network used by Defense 1
                    (Stability Training).
    RobustMLP_v2  : Deeper 256 -> 128 -> 64 -> 2 network with BatchNorm
                    and Dropout, used by Defenses 2, 3, 4.

Both networks output 2 logits (CrossEntropyLoss + targeted IG).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RobustMLP(nn.Module):
    """Compact defense model used by Defense 1 (Stability Training)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RobustMLP_v2(nn.Module):
    """Deeper defense model used by Defenses 2, 3, and 4."""

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
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
