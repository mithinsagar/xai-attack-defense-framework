"""
=====================================================================
XAI Attack and Defense Framework - Confusion Matrix Plots
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from visualization.save_utils import save_and_show


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    cmap: str = "Blues",
) -> None:
    """Render and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_and_show(f"{name}_confusion.png")
