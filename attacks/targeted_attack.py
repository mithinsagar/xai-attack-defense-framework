"""
=====================================================================
XAI Attack and Defense Framework - Attack 4: Targeted Explanation
                                             Attack
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Attack 4 directly targets the top explanation features with a large
perturbation budget. This represents the strongest single-shot attack:
it perturbs 5 features with epsilon = 0.25.

    for i in top_k_explanation_features(x):
        x'[i] = x[i] + epsilon
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import config
from attacks.attack_base import BaseAttack


class TargetedAttack(BaseAttack):
    """Attack 4: aggressive targeted perturbation of top-k explanation features."""

    name = "attack4_targeted"
    epsilon = config.ATTACK_CONFIG["attack4_epsilon"]

    def __init__(
        self,
        feature_names: Sequence[str],
        epsilon: float | None = None,
        top_k: int = config.ATTACK_CONFIG["attack4_top_k"],
    ) -> None:
        super().__init__(feature_names, epsilon)
        self.top_k = top_k

    def perturb(
        self,
        x: np.ndarray,
        model,
        explainer=None,
        top_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        perturbed = x.copy()

        if top_features is None:
            if explainer is None:
                raise ValueError("Either `top_features` or `explainer` must be given.")
            values = np.abs(np.asarray(explainer.shap_values(x.reshape(1, -1))))
            values = values.reshape(-1)
            top_idx = np.argsort(values)[-self.top_k:][::-1]
        else:
            top_idx = [self.feature_names.index(f) for f in top_features[: self.top_k]]

        for idx in top_idx:
            perturbed[idx] += self.epsilon
        return perturbed
