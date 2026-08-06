"""
=====================================================================
XAI Attack and Defense Framework - Model Factory
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Central factory for instantiating any model in the project by string key.
Used by training scripts and by the pipeline orchestrator so that the
mapping from configuration to concrete class lives in a single place.
"""

from __future__ import annotations

from typing import Any

from models.logistic_model import build_logistic_model
from models.mlp import MLP, MLP2Class, MLPSmall
from models.robust_mlp import RobustMLP, RobustMLP_v2
from models.xgboost_model import build_xgboost_model


def build_model(key: str, input_dim: int | None = None, **kwargs: Any):
    """
    Build and return a model by string key.

    Parameters
    ----------
    key : str
        One of {"logistic", "xgboost", "mlp", "mlp2", "mlp_small",
                "robust_mlp", "robust_mlp_v2"}.
    input_dim : int, optional
        Required for all PyTorch models.
    kwargs : dict
        Extra keyword arguments forwarded to the underlying constructor.
    """
    key = key.lower()

    if key == "logistic":
        return build_logistic_model(**kwargs)
    if key == "xgboost":
        return build_xgboost_model(**kwargs)

    if input_dim is None:
        raise ValueError(f"input_dim is required for PyTorch model '{key}'")

    if key == "mlp":
        return MLP(input_dim)
    if key == "mlp2":
        return MLP2Class(input_dim)
    if key == "mlp_small":
        return MLPSmall(input_dim)
    if key == "robust_mlp":
        return RobustMLP(input_dim)
    if key == "robust_mlp_v2":
        return RobustMLP_v2(input_dim)

    raise KeyError(f"Unknown model key: {key}")
