"""
=====================================================================
XAI Attack and Defense Framework - Few-Shot Trainer
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Trains the Prototypical Network encoder using episode-based training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

import config
from fewshot.episode_sampler import create_episode
from models.fewshot_encoder import Encoder, compute_prototypes
from utils.logger import get_logger

LOGGER = get_logger("fewshot.trainer")


def train_prototypical(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int | None = None,
    epochs: int = config.FEWSHOT_CONFIG["epochs"],
    lr: float = config.FEWSHOT_CONFIG["lr"],
) -> Encoder:
    """Train the encoder used by the Prototypical Network."""
    if input_dim is None:
        input_dim = X_train.shape[1]

    encoder = Encoder(input_dim)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(epochs):
        sx, sy, qx, qy = create_episode(X_train, y_train)

        se = encoder(sx)
        qe = encoder(qx)
        prototypes = compute_prototypes(se, sy)

        distances = ((qe.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(2)
        logits = -distances

        loss = ce(logits, qy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0:
            acc = (logits.argmax(1) == qy).float().mean().item()
            LOGGER.info("FewShot | ep %d | loss %.4f | acc %.4f",
                        ep, loss.item(), acc)
    return encoder
