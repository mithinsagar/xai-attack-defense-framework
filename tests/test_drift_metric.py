"""
=====================================================================
XAI Attack and Defense Framework - Tests: Drift Metric
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

import numpy as np

from metrics.drift_metric import (
    compute_all_drift_metrics,
    explanation_drift,
    max_drift,
    rank_correlation_drift,
    topk_drift,
)


def test_zero_drift_when_identical():
    v = np.array([0.1, 0.5, 0.2, 0.9])
    assert explanation_drift(v, v) == 0.0
    assert max_drift(v, v) == 0.0
    assert topk_drift(v, v, k=2) == 0.0
    assert rank_correlation_drift(v, v) == 0.0


def test_positive_drift_when_different():
    a = np.array([0.1, 0.5, 0.2])
    b = np.array([0.4, 0.5, 0.5])
    assert explanation_drift(a, b) > 0.0
    assert max_drift(a, b) > 0.0


def test_topk_drift_returns_mean_of_topk():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.1, 0.2, 0.3])
    # Top-2 abs diffs = [0.3, 0.2] -> mean = 0.25
    assert abs(topk_drift(a, b, k=2) - 0.25) < 1e-9


def test_compute_all_drift_metrics_keys():
    a = np.array([0.1, 0.2])
    b = np.array([0.3, 0.4])
    out = compute_all_drift_metrics(a, b, k=2)
    assert set(out.keys()) == {"Drift", "MaxDrift", "Top2Drift", "RankCorrDrift"}
    for v in out.values():
        assert isinstance(v, float)
