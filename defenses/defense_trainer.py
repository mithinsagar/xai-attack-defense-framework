"""
=====================================================================
XAI Attack and Defense Framework - Defense Trainer Dispatcher
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Convenience function that trains a defense model by name and returns the
fitted network. Used by `training/train_defense_models.py`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from defenses.adversarial_training import train_adversarial
from defenses.explanation_regularization import train_explanation_reg
from defenses.hybrid_defense import train_hybrid
from defenses.stability_training import train_stability


DEFENSE_REGISTRY = {
    "defense1_stability":       train_stability,
    "defense2_adversarial":     train_adversarial,
    "defense3_explanation_reg": train_explanation_reg,
    "defense4_hybrid":          train_hybrid,
}


def train_defense(name: str, **kwargs: Any) -> nn.Module:
    """
    Train a defense by string key.

    Parameters
    ----------
    name : str
        One of the keys in `DEFENSE_REGISTRY`.
    kwargs : dict
        Forwarded to the underlying trainer (X_train, y_train, etc.).
    """
    if name not in DEFENSE_REGISTRY:
        raise KeyError(f"Unknown defense '{name}'. Valid: {list(DEFENSE_REGISTRY)}")
    return DEFENSE_REGISTRY[name](**kwargs)
