"""
=====================================================================
XAI Attack and Defense Framework - Attack 1: Top-Feature Perturbation
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Attack 1 identifies the top-k SHAP features and applies a small (epsilon)
additive perturbation to each. Tests whether the model's explanation is
stable near the decision boundary of the features it "cares about" most.

    for i in top_k_SHAP_features(x):
        x'[i] = x[i] + epsilon
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import config
from attacks.attack_base import BaseAttack


class TopFeatureAttack(BaseAttack):
    """Attack 1: perturb the top-k SHAP features by +epsilon."""

    name = "attack1_top_feature"
    epsilon = config.ATTACK_CONFIG["attack1_epsilon"]

    def __init__(
        self,
        feature_names: Sequence[str],
        epsilon: float | None = None,
        top_k: int = config.ATTACK_CONFIG["attack1_top_k"],
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
        """
        Parameters
        ----------
        x : np.ndarray
            Single sample (1-D vector of shape (d,)).
        model : object
            Trained classifier (unused here but kept for interface parity).
        explainer : object, optional
            SHAP explainer used to identify top features if not supplied.
        top_features : Sequence[str], optional
            Pre-computed top-k feature names.
        """
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
