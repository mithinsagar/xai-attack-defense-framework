"""
=====================================================================
XAI Attack and Defense Framework - Attack 2: Prediction-Stable
                                             Explanation Manipulation
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Attack 2 perturbs low-importance features. The model prediction is
expected to stay unchanged, yet the explanation still shifts significantly.
This is arguably the most dangerous attack type because it changes
features the analyst would not expect to matter.

    for i in bottom_k_SHAP_features(x):
        x'[i] = x[i] + epsilon
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import config
from attacks.attack_base import BaseAttack


class PredictionStableAttack(BaseAttack):
    """Attack 2: perturb the bottom-k SHAP features by +epsilon."""

    name = "attack2_prediction_stable"
    epsilon = config.ATTACK_CONFIG["attack2_epsilon"]

    def __init__(
        self,
        feature_names: Sequence[str],
        epsilon: float | None = None,
        low_k: int = config.ATTACK_CONFIG["attack2_low_k"],
    ) -> None:
        super().__init__(feature_names, epsilon)
        self.low_k = low_k

    def perturb(
        self,
        x: np.ndarray,
        model,
        explainer=None,
        low_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        perturbed = x.copy()

        if low_features is None:
            if explainer is None:
                raise ValueError("Either `low_features` or `explainer` must be given.")
            values = np.abs(np.asarray(explainer.shap_values(x.reshape(1, -1))))
            values = values.reshape(-1)
            low_idx = np.argsort(values)[: self.low_k]
        else:
            low_idx = [self.feature_names.index(f) for f in low_features[: self.low_k]]

        for idx in low_idx:
            perturbed[idx] += self.epsilon
        return perturbed
