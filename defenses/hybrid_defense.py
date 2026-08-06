"""
=====================================================================
XAI Attack and Defense Framework - Defense 4: Hybrid Defense (BEST)
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Defense 4 combines Stability, Adversarial and Smoothness loss terms into
a single training objective:

    L_hybrid = CE(f(x), y)
             + w_stability * mean|f(x) - f(x + N(0, sigma_s^2))|
             + w_adv       * CE(f(x + N(0, sigma_l^2)), y)
             + w_smooth    * mean|f(x + N(0, sigma_s^2)) - f(x + N(0, sigma_l^2))|

Empirically achieves ~91.1 percent explanation drift reduction, the best
of all four defenses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

import config
from models.robust_mlp import RobustMLP_v2
from utils.logger import get_logger

LOGGER = get_logger("defenses.hybrid")


def train_hybrid(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int | None = None,
    epochs: int = config.DEFENSE_CONFIG["defense4"]["epochs"],
    lr: float = config.DEFENSE_CONFIG["defense4"]["lr"],
    small_noise: float = config.DEFENSE_CONFIG["defense4"]["small_noise"],
    large_noise: float = config.DEFENSE_CONFIG["defense4"]["large_noise"],
    stability_weight: float = config.DEFENSE_CONFIG["defense4"]["stability_weight"],
    adv_weight: float = config.DEFENSE_CONFIG["defense4"]["adv_weight"],
    smooth_weight: float = config.DEFENSE_CONFIG["defense4"]["smooth_weight"],
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

        n_small = torch.randn_like(X_train) * small_noise
        out_small = model(X_train + n_small)
        stability = torch.mean(torch.abs(out - out_small))

        n_large = torch.randn_like(X_train) * large_noise
        out_large = model(X_train + n_large)
        adv = loss_fn(out_large, y_train)

        smooth = torch.mean(torch.abs(out_small - out_large))

        loss = (
            ce
            + stability_weight * stability
            + adv_weight * adv
            + smooth_weight * smooth
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_acc = (model(X_val).argmax(1) == y_val).float().mean()
            LOGGER.info(
                "Def4 | epoch %d | loss %.4f | val_acc %.4f",
                epoch, loss.item(), val_acc.item(),
            )
    return model
