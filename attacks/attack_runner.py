"""
=====================================================================
XAI Attack and Defense Framework - Attack Runner
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Runs all four attacks against every (dataset, model) combination and
writes an aggregated `attack_results.csv` under `results/`.

Columns:
    Dataset  |  Model  |  Attack  |  Drift  |  MaxDrift  |  Top5Drift
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap

import config
from attacks.gradient_attack import GradientAttack
from attacks.prediction_stable_attack import PredictionStableAttack
from attacks.targeted_attack import TargetedAttack
from attacks.top_feature_attack import TopFeatureAttack
from explainers.integrated_gradients import compute_ig_attributions
from metrics.drift_metric import compute_all_drift_metrics
from models.mlp import MLP
from utils.io_utils import load_numpy, load_pickle, load_torch_model
from utils.logger import get_logger

LOGGER = get_logger("attacks.runner")


def _tree_shap_abs(explainer, x: np.ndarray) -> np.ndarray:
    values = np.asarray(explainer.shap_values(x.reshape(1, -1)))
    return np.abs(values).reshape(-1)


def _ig_abs(model, x: np.ndarray) -> np.ndarray:
    return compute_ig_attributions(model, x)


def _run_for_xgb(key: str) -> List[Dict[str, Any]]:
    X_test = load_numpy(config.PROCESSED_DATA_DIR / f"X_test_{key}.npy")
    features = load_pickle(config.PROCESSED_DATA_DIR / f"feature_{key}.pkl")
    xgb = load_pickle(config.BASE_MODELS_DIR / f"xgb_{key}.pkl")

    sample = X_test[0]
    explainer = shap.TreeExplainer(xgb)
    attr_before = _tree_shap_abs(explainer, sample)

    attacks = [
        TopFeatureAttack(features),
        PredictionStableAttack(features),
        TargetedAttack(features),
    ]
    rows: List[Dict[str, Any]] = []
    for atk in attacks:
        perturbed = atk.perturb(sample, xgb, explainer=explainer)
        attr_after = _tree_shap_abs(explainer, perturbed)
        metrics = compute_all_drift_metrics(attr_before, attr_after)
        rows.append({
            "Dataset": key, "Model": "XGBoost", "Attack": atk.name, **metrics,
        })
    return rows


def _run_for_mlp(key: str) -> List[Dict[str, Any]]:
    X_test = load_numpy(config.PROCESSED_DATA_DIR / f"X_test_{key}.npy")
    features = load_pickle(config.PROCESSED_DATA_DIR / f"feature_{key}.pkl")

    model = MLP(X_test.shape[1])
    model = load_torch_model(model, config.BASE_MODELS_DIR / f"nn_{key}.pt")
    model.eval()

    sample = X_test[0]
    attr_before = _ig_abs(model, sample)

    attacks = [
        TopFeatureAttack(features),
        PredictionStableAttack(features),
        GradientAttack(features),
        TargetedAttack(features),
    ]
    rows: List[Dict[str, Any]] = []
    for atk in attacks:
        perturbed = atk.perturb(sample, model)
        attr_after = _ig_abs(model, perturbed)
        metrics = compute_all_drift_metrics(attr_before, attr_after)
        rows.append({
            "Dataset": key, "Model": "Neural", "Attack": atk.name, **metrics,
        })
    return rows


def run_all_attacks() -> pd.DataFrame:
    """Run all attacks for all (dataset, model) combinations."""
    config.ensure_directories()
    all_rows: List[Dict[str, Any]] = []
    for key in config.DATASETS:
        LOGGER.info("Running attacks on %s", key)
        try:
            all_rows.extend(_run_for_xgb(key))
        except FileNotFoundError as e:
            LOGGER.warning("XGBoost artefacts missing for %s: %s", key, e)
        try:
            all_rows.extend(_run_for_mlp(key))
        except FileNotFoundError as e:
            LOGGER.warning("Neural model missing for %s: %s", key, e)

    df = pd.DataFrame(all_rows)
    out = config.RESULTS_DIR / "attack_results.csv"
    df.to_csv(out, index=False)
    LOGGER.info("Attack results saved -> %s", out)
    return df
