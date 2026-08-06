"""
=====================================================================
XAI Attack and Defense Framework - Prototypical Network
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Minimal Prototypical Network wrapper that combines the encoder and the
prototype-distance classifier.

Reference: Snell et al., "Prototypical Networks for Few-shot Learning",
NeurIPS 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.fewshot_encoder import Encoder, compute_prototypes


class PrototypicalNetwork(nn.Module):
    """Prototypical Network built on top of a shared encoder."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        se = self.encoder(support_x)
        qe = self.encoder(query_x)
        prototypes = compute_prototypes(se, support_y)
        distances = ((qe.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(2)
        return -distances  # logits
