"""
=====================================================================
XAI Attack and Defense Framework - SHAP Explanations
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Thin wrappers around SHAP's TreeExplainer (for XGBoost) and KernelExplainer
(for Logistic Regression), plus helpers to convert SHAP values into
a sorted feature-importance DataFrame.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import shap

from utils.helpers import feature_importance_table


def tree_shap_values(model, X: np.ndarray) -> np.ndarray:
    """Compute exact SHAP values for a tree-based model (e.g. XGBoost)."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    return np.asarray(values)


def kernel_shap_values(
    model, X: np.ndarray, background: np.ndarray, nsamples: str | int = "auto"
) -> np.ndarray:
    """Compute approximate SHAP values via KernelExplainer (model-agnostic)."""
    explainer = shap.KernelExplainer(model.predict_proba, background)
    values = explainer.shap_values(X, nsamples=nsamples)
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values)


def global_shap_importance(
    shap_values: np.ndarray, feature_names: Sequence[str]
) -> pd.DataFrame:
    """
    Aggregate per-sample SHAP values into a global feature importance
    ranking by taking the mean absolute value per feature.
    """
    if shap_values.ndim == 1:
        importance = np.abs(shap_values)
    else:
        importance = np.abs(shap_values).mean(axis=0)
    return feature_importance_table(feature_names, importance)
