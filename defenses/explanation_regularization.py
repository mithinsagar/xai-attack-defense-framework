"""
=====================================================================
XAI Attack and Defense Framework - Defense 3: Explanation
                                             Regularization
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Defense 3 directly regularises the explanation itself. During training,
Integrated Gradients attributions are computed for the clean and noisy
inputs and the L1 distance between them is added as a loss term:

    L = CE(f(x), y) + lambda_3 * mean|IG(x) - IG(x + N(0, sigma^2))|

Because IG is expensive, only `ig_batch_size` samples participate in the
explanation loss per epoch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients

import config
from models.robust_mlp import RobustMLP_v2
from utils.logger import get_logger

LOGGER = get_logger("defenses.explanation_reg")


def train_explanation_reg(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int | None = None,
    epochs: int = config.DEFENSE_CONFIG["defense3"]["epochs"],
    lr: float = config.DEFENSE_CONFIG["defense3"]["lr"],
    noise_sigma: float = config.DEFENSE_CONFIG["defense3"]["noise_sigma"],
    explain_weight: float = config.DEFENSE_CONFIG["defense3"]["explain_weight"],
    ig_batch_size: int = config.DEFENSE_CONFIG["defense3"]["ig_batch_size"],
) -> nn.Module:
    if input_dim is None:
        input_dim = X_train.shape[1]

    model = RobustMLP_v2(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    ig = IntegratedGradients(model)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(X_train)
        ce = loss_fn(out, y_train)

        noise = torch.randn_like(X_train) * noise_sigma
        attr_clean = ig.attribute(X_train[:ig_batch_size], target=0)
        attr_noisy = ig.attribute((X_train + noise)[:ig_batch_size], target=0)
        explain = torch.mean(torch.abs(attr_clean - attr_noisy))

        loss = ce + explain_weight * explain
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            LOGGER.info("Def3 | epoch %d | loss %.4f", epoch, loss.item())
    return model
