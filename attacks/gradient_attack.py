"""
=====================================================================
XAI Attack and Defense Framework - Attack 3: Gradient-Based
                                             Explanation Attack
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Attack 3 exploits the neural network's own gradients to find the most
sensitive perturbation direction:

    1. Forward pass  :  y_hat = f(x)
    2. Backward pass :  g = grad_x f(x)
    3. Identify top-k features by |g|
    4. Perturb        :  x'[i] = x[i] + epsilon  for top-k features

Two variants are provided:
    * `gradient_topk_perturb`  - additive perturbation of top-k features
    * `fgsm_sign_perturb`      - FGSM-style sign attack on the whole vector
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

import config
from attacks.attack_base import BaseAttack
from utils.helpers import to_numpy, to_tensor


class GradientAttack(BaseAttack):
    """Attack 3: gradient-guided perturbation of the top-k input features."""

    name = "attack3_gradient"
    epsilon = config.ATTACK_CONFIG["attack3_epsilon"]

    def __init__(
        self,
        feature_names: Sequence[str],
        epsilon: float | None = None,
        top_k: int = config.ATTACK_CONFIG["attack3_top_k"],
    ) -> None:
        super().__init__(feature_names, epsilon)
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
        grad = self._compute_grad(model, x)
        top_idx = np.argsort(np.abs(grad))[-self.top_k:][::-1]
        perturbed = x.copy()
        for idx in top_idx:
            perturbed[idx] += self.epsilon
        return perturbed

    def fgsm_sign_perturb(self, model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
        """Full-vector FGSM-style sign perturbation."""
        grad = self._compute_grad(model, x)
        return x + self.epsilon * np.sign(grad)
