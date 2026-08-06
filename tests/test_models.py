"""
=====================================================================
XAI Attack and Defense Framework - Tests: Models
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

import numpy as np
import torch

from models.mlp import MLP, MLP2Class, MLPSmall
from models.model_factory import build_model
from models.robust_mlp import RobustMLP, RobustMLP_v2


def test_mlp_forward_shape():
    m = MLP(input_dim=10)
    x = torch.randn(4, 10)
    out = m(x)
    assert out.shape == (4, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_mlp2class_forward_shape():
    m = MLP2Class(input_dim=8)
    x = torch.randn(6, 8)
    out = m(x)
    assert out.shape == (6, 2)


def test_robust_mlp_forward_shape():
    m = RobustMLP(input_dim=5)
    x = torch.randn(3, 5)
    assert m(x).shape == (3, 2)


def test_robust_mlp_v2_forward_shape():
    m = RobustMLP_v2(input_dim=12)
    x = torch.randn(2, 12)
    assert m(x).shape == (2, 2)


def test_model_factory_keys():
    for key in ("logistic", "xgboost"):
        m = build_model(key)
        assert m is not None

    for key in ("mlp", "mlp2", "mlp_small", "robust_mlp", "robust_mlp_v2"):
        m = build_model(key, input_dim=8)
        assert isinstance(m, torch.nn.Module)


def test_mlp_small_forward_shape():
    m = MLPSmall(input_dim=6)
    x = torch.randn(1, 6)
    assert m(x).shape == (1, 1)
