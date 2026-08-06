"""
=====================================================================
XAI Attack and Defense Framework - Tests: Attacks
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

import numpy as np
import torch

from attacks.gradient_attack import GradientAttack
from attacks.prediction_stable_attack import PredictionStableAttack
from attacks.targeted_attack import TargetedAttack
from attacks.top_feature_attack import TopFeatureAttack
from models.mlp import MLP


FEATURES = [f"f{i}" for i in range(6)]


def _sample():
    rng = np.random.default_rng(0)
    return rng.normal(size=6)


def test_top_feature_attack_changes_top_k_features():
    atk = TopFeatureAttack(FEATURES, epsilon=0.1, top_k=2)
    x = _sample()
    perturbed = atk.perturb(x, model=None, top_features=["f3", "f1"])
    assert perturbed[3] == x[3] + 0.1
    assert perturbed[1] == x[1] + 0.1


def test_prediction_stable_attack_changes_low_features():
    atk = PredictionStableAttack(FEATURES, epsilon=0.05, low_k=1)
    x = _sample()
    perturbed = atk.perturb(x, model=None, low_features=["f5"])
    assert perturbed[5] == x[5] + 0.05


def test_targeted_attack_epsilon_larger():
    atk = TargetedAttack(FEATURES, epsilon=0.25, top_k=1)
    x = _sample()
    perturbed = atk.perturb(x, model=None, top_features=["f0"])
    assert perturbed[0] == x[0] + 0.25


def test_gradient_attack_returns_perturbed_vector():
    model = MLP(input_dim=6)
    atk = GradientAttack(FEATURES, epsilon=0.05, top_k=2)
    x = _sample()
    perturbed = atk.perturb(x, model)
    assert perturbed.shape == x.shape
    # At least top_k features should differ
    diff = np.abs(perturbed - x)
    assert (diff > 0).sum() >= 2
