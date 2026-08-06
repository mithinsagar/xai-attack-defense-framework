"""
=====================================================================
XAI Attack and Defense Framework - Few-Shot Prototypical Encoder
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Feature encoder used by the Prototypical Network few-shot classifier.

Architecture (matches Phase 5 of the original notebook):
    Input (d)
        -> Linear(d, 128) + ReLU
        -> Linear(128, 64)

Prototypes are class-wise mean embeddings; classification uses negative
squared Euclidean distance as logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class Encoder(nn.Module):
    """Two-layer MLP encoder mapping inputs to a 64-dim embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = config.FEWSHOT_CONFIG["encoder_hidden"],
        output_dim: int = config.FEWSHOT_CONFIG["encoder_output"],
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_prototypes(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute class-wise mean embeddings (prototypes)."""
    classes = torch.unique(labels)
    return torch.stack([embeddings[labels == c].mean(0) for c in classes])
