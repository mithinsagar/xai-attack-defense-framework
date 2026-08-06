"""
=====================================================================
XAI Attack and Defense Framework - LIME Explanations
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Wrapper around LIME's `LimeTabularExplainer` that returns per-feature
importance for a single instance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from lime.lime_tabular import LimeTabularExplainer


def build_lime_explainer(
    training_data: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str],
) -> LimeTabularExplainer:
    """Instantiate a LimeTabularExplainer for tabular classification."""
    return LimeTabularExplainer(
        training_data=training_data,
        feature_names=list(feature_names),
        class_names=list(class_names),
        mode="classification",
    )


def explain_instance(
    explainer: LimeTabularExplainer,
    instance: np.ndarray,
    predict_fn,
    num_features: int = 10,
):
    """Generate a LIME explanation for a single instance."""
    return explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=num_features,
    )
