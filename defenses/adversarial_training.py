"""
=====================================================================
XAI Attack and Defense Framework - Defense 2: Adversarial Training
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Defense 2 trains on adversarially perturbed examples alongside clean
examples:

    L = CE(f(x), y) + lambda_2 * CE(f(x + delta_adv), y)

with delta_adv ~ N(0, sigma^2) and sigma = 0.05.

Focuses on prediction correctness under perturbation; empirically
improves prediction robustness more than explanation robustness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

import config
from models.robust_mlp import RobustMLP_v2
from utils.logger import get_logger

LOGGER = get_logger("defenses.adversarial")


def train_adversarial(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int | None = None,
    epochs: int = config.DEFENSE_CONFIG["defense2"]["epochs"],
    lr: float = config.DEFENSE_CONFIG["defense2"]["lr"],
    noise_sigma: float = config.DEFENSE_CONFIG["defense2"]["noise_sigma"],
    adv_weight: float = config.DEFENSE_CONFIG["defense2"]["adv_weight"],
) -> nn.Module:
    if input_dim is None:
        input_dim = X_train.shape[1]

    model = RobustMLP_v2(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(X_train)
        ce = loss_fn(out, y_train)

        noise = torch.randn_like(X_train) * noise_sigma
        adv_out = model(X_train + noise)
        adv_ce = loss_fn(adv_out, y_train)

        loss = ce + adv_weight * adv_ce
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                val_acc = (val_out.argmax(1) == y_val).float().mean()
            LOGGER.info(
                "Def2 | epoch %d | train %.4f | val_acc %.4f",
                epoch, loss.item(), val_acc.item(),
            )
    return model
