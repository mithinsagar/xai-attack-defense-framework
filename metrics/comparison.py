"""
=====================================================================
XAI Attack and Defense Framework - Result Comparison Tables
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Helpers for building the summary tables shown in the technical report:
    - Attack effectiveness ranking
    - Defense effectiveness ranking
    - Few-shot vs full model comparison
"""

from __future__ import annotations

import pandas as pd


def attack_ranking(df: pd.DataFrame, drift_col: str = "Drift") -> pd.DataFrame:
    """Rank attacks by mean explanation drift across models and datasets."""
    if "Attack" not in df.columns:
        raise KeyError("Expected a column named 'Attack' in df.")
    return (
        df.groupby("Attack")[drift_col]
        .mean()
        .reset_index()
        .sort_values(drift_col, ascending=False)
        .reset_index(drop=True)
    )


def model_ranking(df: pd.DataFrame, drift_col: str = "Drift") -> pd.DataFrame:
    """Rank models by mean explanation drift across attacks and datasets."""
    if "Model" not in df.columns:
        raise KeyError("Expected a column named 'Model' in df.")
    return (
        df.groupby("Model")[drift_col]
        .mean()
        .reset_index()
        .sort_values(drift_col, ascending=False)
        .reset_index(drop=True)
    )


def dataset_ranking(df: pd.DataFrame, drift_col: str = "Drift") -> pd.DataFrame:
    """Rank datasets by mean explanation drift."""
    if "Dataset" not in df.columns:
        raise KeyError("Expected a column named 'Dataset' in df.")
    return (
        df.groupby("Dataset")[drift_col]
        .mean()
        .reset_index()
        .sort_values(drift_col, ascending=False)
        .reset_index(drop=True)
    )


def defense_reduction_table(
    baseline_drift: float,
    defense_drifts: dict[str, float],
) -> pd.DataFrame:
    """Compute the percentage drift reduction achieved by each defense."""
    rows = [{"Model": "Baseline", "Drift": baseline_drift, "Reduction (%)": 0.0}]
    for name, drift in defense_drifts.items():
        reduction = 100.0 * (baseline_drift - drift) / max(baseline_drift, 1e-12)
        rows.append({"Model": name, "Drift": drift, "Reduction (%)": reduction})
    return pd.DataFrame(rows)
