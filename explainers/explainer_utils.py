"""
=====================================================================
XAI Attack and Defense Framework - Explainer Utilities
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Phase 3 orchestration: generate SHAP, LIME, IG, and permutation-importance
explanations for all trained base models on the three cybersecurity datasets.
Persists per-dataset importance tables to `results/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from explainers.integrated_gradients import ig_importance_table
from explainers.permutation_importance import permutation_importance_table
from explainers.shap_explainer import global_shap_importance, tree_shap_values
from models.mlp import MLP
from utils.io_utils import load_numpy, load_pickle, load_torch_model
from utils.logger import get_logger

LOGGER = get_logger("explainers")


def _shap_for_xgb(key: str) -> None:
    X_test = load_numpy(config.PROCESSED_DATA_DIR / f"X_test_{key}.npy")
    features = load_pickle(config.PROCESSED_DATA_DIR / f"feature_{key}.pkl")
    xgb = load_pickle(config.BASE_MODELS_DIR / f"xgb_{key}.pkl")

    sample = X_test[:500] if len(X_test) >= 500 else X_test
    values = tree_shap_values(xgb, sample)
    table = global_shap_importance(values, features)
    out = config.RESULTS_DIR / f"shap_xgb_{key}.csv"
    table.to_csv(out, index=False)
    LOGGER.info("SHAP importance saved -> %s", out)


def _perm_for_xgb(key: str) -> None:
    X_test = load_numpy(config.PROCESSED_DATA_DIR / f"X_test_{key}.npy")
    y_test = load_numpy(config.PROCESSED_DATA_DIR / f"y_test_{key}.npy", allow_pickle=True)
    features = load_pickle(config.PROCESSED_DATA_DIR / f"feature_{key}.pkl")
    xgb = load_pickle(config.BASE_MODELS_DIR / f"xgb_{key}.pkl")

    table = permutation_importance_table(xgb, X_test, y_test, features, n_repeats=10)
    out = config.RESULTS_DIR / f"perm_xgb_{key}.csv"
    table.to_csv(out, index=False)
    LOGGER.info("Permutation importance saved -> %s", out)


def _ig_for_mlp(key: str) -> None:
    X_test = load_numpy(config.PROCESSED_DATA_DIR / f"X_test_{key}.npy")
    features = load_pickle(config.PROCESSED_DATA_DIR / f"feature_{key}.pkl")

    model = MLP(X_test.shape[1])
    model = load_torch_model(model, config.BASE_MODELS_DIR / f"nn_{key}.pt")

    table = ig_importance_table(model, X_test[0], features)
    out = config.RESULTS_DIR / f"ig_mlp_{key}.csv"
    table.to_csv(out, index=False)
    LOGGER.info("Integrated Gradients importance saved -> %s", out)


def generate_all_explanations() -> None:
    """Phase 3 entrypoint: run all four explanation methods for all datasets."""
    config.ensure_directories()
    for key in config.DATASETS:
        LOGGER.info("Generating explanations for dataset: %s", key)
        try:
            _shap_for_xgb(key)
            _perm_for_xgb(key)
            _ig_for_mlp(key)
        except FileNotFoundError as e:
            LOGGER.warning("Skipping %s: %s", key, e)
