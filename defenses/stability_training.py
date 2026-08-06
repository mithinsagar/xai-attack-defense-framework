"""
=====================================================================
XAI Attack and Defense Framework - Defense 1: Stability Training
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Defense 1 adds Gaussian noise to inputs during training and penalises the
absolute difference between clean and noisy outputs:

    L = CE(f(x), y) + lambda_1 * mean|f(x) - f(x + N(0, sigma^2))|

By forcing the model to produce similar outputs for clean and slightly
perturbed inputs, the learned representations become smoother, which
transitively stabilises explanations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

import config
from models.robust_mlp import RobustMLP
from utils.logger import get_logger

LOGGER = get_logger("defenses.stability")


def train_stability(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int | None = None,
    epochs: int = config.DEFENSE_CONFIG["defense1"]["epochs"],
    lr: float = config.DEFENSE_CONFIG["defense1"]["lr"],
    noise_sigma: float = config.DEFENSE_CONFIG["defense1"]["noise_sigma"],
    stability_weight: float = config.DEFENSE_CONFIG["defense1"]["stability_weight"],
) -> nn.Module:
    """
    Train a RobustMLP using stability training.

    Returns
    -------
    model : torch.nn.Module
        The trained defense model.
    """
    if input_dim is None:
        input_dim = X_train.shape[1]

    model = RobustMLP(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(X_train)
        ce = loss_fn(out, y_train)

        noise = torch.randn_like(X_train) * noise_sigma
        out_noisy = model(X_train + noise)
        stability = torch.mean(torch.abs(out - out_noisy))

        loss = ce + stability_weight * stability
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                val_loss = loss_fn(val_out, y_val)
                val_acc = (val_out.argmax(1) == y_val).float().mean()
            LOGGER.info(
                "Def1 | epoch %d | train %.4f | val %.4f | val_acc %.4f",
                epoch, loss.item(), val_loss.item(), val_acc.item(),
            )
    return model
