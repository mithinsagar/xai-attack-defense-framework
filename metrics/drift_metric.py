"""
=====================================================================
XAI Attack and Defense Framework - Explanation Drift Metric
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Implementation of the Explanation Drift metric introduced in the
technical report (Section 9):

    D(x, epsilon) = (1 / d) * sum_i |E_i(x) - E_i(x + delta)|

Also provides three complementary metrics:
    * MaxDrift              : max_i |E_i(x) - E_i(x + delta)|
    * Top-k Drift           : mean drift over the top-k largest per-feature drifts
    * Rank Correlation Drift: 1 - |Spearman(rank_before, rank_after)|
                              (higher = more manipulation)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import spearmanr


def explanation_drift(before: np.ndarray, after: np.ndarray) -> float:
    """Mean absolute drift across all features."""
    before = np.asarray(before).reshape(-1)
    after = np.asarray(after).reshape(-1)
    return float(np.mean(np.abs(before - after)))


def max_drift(before: np.ndarray, after: np.ndarray) -> float:
    """Maximum per-feature absolute drift."""
    before = np.asarray(before).reshape(-1)
    after = np.asarray(after).reshape(-1)
    return float(np.max(np.abs(before - after)))


def topk_drift(before: np.ndarray, after: np.ndarray, k: int = 5) -> float:
    """Mean drift of the k features with the largest individual drifts."""
    before = np.asarray(before).reshape(-1)
    after = np.asarray(after).reshape(-1)
    diffs = np.sort(np.abs(before - after))[-k:]
    return float(np.mean(diffs))


def rank_correlation_drift(before: np.ndarray, after: np.ndarray) -> float:
    """1 - |Spearman correlation of ranked feature importances|."""
    before = np.asarray(before).reshape(-1)
    after = np.asarray(after).reshape(-1)
    if before.size < 2:
        return 0.0
    corr, _ = spearmanr(before, after)
    if np.isnan(corr):
        return 0.0
    return float(1.0 - abs(corr))


def compute_all_drift_metrics(
    before: np.ndarray, after: np.ndarray, k: int = 5
) -> Dict[str, float]:
    """Compute all four drift metrics in one shot."""
    return {
        "Drift": explanation_drift(before, after),
        "MaxDrift": max_drift(before, after),
        f"Top{k}Drift": topk_drift(before, after, k=k),
        "RankCorrDrift": rank_correlation_drift(before, after),
    }
