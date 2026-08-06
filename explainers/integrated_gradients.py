"""
=====================================================================
XAI Attack and Defense Framework - Integrated Gradients (Captum)
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Utility wrapper around Captum's IntegratedGradients. Provides absolute
per-feature attributions for a single input sample or a batch, plus a
convenience `ig_importance_table` that returns a sorted DataFrame.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from utils.helpers import feature_importance_table, to_numpy, to_tensor


def compute_ig_attributions(
    model: torch.nn.Module,
    x: np.ndarray | torch.Tensor,
    target: int | None = None,
    baseline: torch.Tensor | None = None,
    return_delta: bool = False,
):
    """
    Compute Integrated Gradients attributions for `x`.

    Returns
    -------
    attributions : np.ndarray
        Absolute per-feature attribution values.
    delta : float (optional)
        Convergence delta returned by Captum, if `return_delta=True`.
    """
    if isinstance(x, np.ndarray):
        x = to_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(0)

    model.eval()
    ig = IntegratedGradients(model)

    if return_delta:
        attr, delta = ig.attribute(
            x, target=target, baselines=baseline,
            return_convergence_delta=True,
        )
        return np.abs(to_numpy(attr)[0]), float(delta)
    else:
        attr = ig.attribute(x, target=target, baselines=baseline)
        return np.abs(to_numpy(attr)[0])


def ig_importance_table(
    model: torch.nn.Module,
    x: np.ndarray,
    feature_names: Sequence[str],
    target: int | None = None,
) -> pd.DataFrame:
    """Return a sorted (Feature, Importance) DataFrame for a single sample."""
    attr = compute_ig_attributions(model, x, target=target)
    return feature_importance_table(feature_names, attr)
