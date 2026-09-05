"""
=====================================================================
XAI Attack and Defense Framework - Attack 5: Iterative Gradient
                                             Explanation Attack
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Attack 5 extends Attack 3's single gradient step into an iterative,
BIM/I-FGSM-style procedure: repeatedly take small gradient-sign steps
on the top-k most sensitive features, projecting back into the
epsilon L-infinity ball after every step. Iterating lets the attack
climb further along the drift-maximising direction than a single
gradient step can reach, while still respecting the same overall
perturbation budget:

    x_0          = x
    g_t          = grad_x f(x_t)
    x_{t+1}[i]   = clip(x_t[i] + alpha * sign(g_t[i]), x[i]-eps, x[i]+eps)
                   for i in top_k(|g_t|), repeated over `num_steps` iterations
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

import config
from attacks.attack_base import BaseAttack
from utils.helpers import to_numpy, to_tensor


class IterativeGradientAttack(BaseAttack):
    """Attack 5: iterative (BIM/I-FGSM-style) gradient-guided perturbation."""

    name = "attack5_iterative_gradient"
    epsilon = config.ATTACK_CONFIG["attack5_epsilon"]

    def __init__(
        self,
        feature_names: Sequence[str],
        epsilon: float | None = None,
        alpha: float = config.ATTACK_CONFIG["attack5_alpha"],
        num_steps: int = config.ATTACK_CONFIG["attack5_steps"],
        top_k: int = config.ATTACK_CONFIG["attack5_top_k"],
    ) -> None:
        super().__init__(feature_names, epsilon)
        self.alpha = alpha
        self.num_steps = num_steps
        self.top_k = top_k

    def _compute_grad(
        self, model: torch.nn.Module, x: np.ndarray
    ) -> np.ndarray:
        x_t = to_tensor(x.reshape(1, -1))
        x_t.requires_grad_(True)
        model.eval()
        out = model(x_t)
        out.sum().backward()
        return to_numpy(x_t.grad)[0]

    def perturb(self, x: np.ndarray, model, explainer=None) -> np.ndarray:
        lo, hi = x - self.epsilon, x + self.epsilon
        perturbed = x.copy()

        for _ in range(self.num_steps):
            grad = self._compute_grad(model, perturbed)
            top_idx = np.argsort(np.abs(grad))[-self.top_k:][::-1]
            for idx in top_idx:
                perturbed[idx] += self.alpha * np.sign(grad[idx])
            perturbed = np.clip(perturbed, lo, hi)

        return perturbed
