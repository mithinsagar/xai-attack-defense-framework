"""
=====================================================================
XAI Attack and Defense Framework - Tests: Defense Trainers
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

import torch

from defenses.adversarial_training import train_adversarial
from defenses.hybrid_defense import train_hybrid
from defenses.stability_training import train_stability


def _tiny_dataset():
    torch.manual_seed(0)
    X = torch.randn(64, 8)
    y = torch.randint(0, 2, (64,))
    return X, y


def test_stability_defense_runs_end_to_end():
    X, y = _tiny_dataset()
    model = train_stability(X, y, X, y, epochs=2)
    assert model is not None
    with torch.no_grad():
        out = model(X)
    assert out.shape == (64, 2)


def test_adversarial_defense_runs_end_to_end():
    X, y = _tiny_dataset()
    model = train_adversarial(X, y, X, y, epochs=2)
    assert model is not None
    with torch.no_grad():
        out = model(X)
    assert out.shape == (64, 2)


def test_hybrid_defense_runs_end_to_end():
    X, y = _tiny_dataset()
    model = train_hybrid(X, y, X, y, epochs=2)
    assert model is not None
    with torch.no_grad():
        out = model(X)
    assert out.shape == (64, 2)
