"""
=====================================================================
XAI Attack and Defense Framework - Classification Evaluation
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Standard classification metrics for base models: accuracy, F1, confusion
matrix. Kept separate from drift metrics so the two concerns don't
cross-contaminate.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_sklearn(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate a scikit-learn / XGBoost model on the test set."""
    preds = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }


def evaluate_torch(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a PyTorch classifier that outputs sigmoid probabilities."""
    model.eval()
    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()
    preds = (probs.reshape(-1) > threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the confusion matrix as a NumPy array."""
    return confusion_matrix(y_true, y_pred)


def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return sklearn's classification_report as a formatted string."""
    return classification_report(y_true, y_pred)
