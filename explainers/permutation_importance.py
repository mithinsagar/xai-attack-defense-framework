"""
=====================================================================
XAI Attack and Defense Framework - Permutation Importance
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Thin wrapper around sklearn.inspection.permutation_importance that
returns a sorted (Feature, Importance) DataFrame.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config
from utils.helpers import feature_importance_table


def permutation_importance_table(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 10,
    random_state: int = config.RANDOM_SEED,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute permutation importance and return a sorted DataFrame."""
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return feature_importance_table(feature_names, result.importances_mean)
