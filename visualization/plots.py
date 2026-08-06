"""
=====================================================================
XAI Attack and Defense Framework - Common Plot Helpers
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.constants import MODERN_COLORS, SEABORN_CONTEXT, SEABORN_STYLE
from visualization.save_utils import save_and_show


def setup_style() -> None:
    sns.set_style(SEABORN_STYLE)
    sns.set_context(SEABORN_CONTEXT)


def plot_label_distribution(y: Sequence[int], dataset_name: str) -> None:
    """Count-plot of class labels for a dataset."""
    setup_style()
    plt.figure(figsize=(7, 5))
    sns.countplot(x=list(y), palette=list(MODERN_COLORS))
    plt.title(f"{dataset_name} - Label Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Count")
    save_and_show(f"{dataset_name.lower()}_distribution.png")


def plot_topn_importance(
    table: pd.DataFrame,
    top_n: int = 10,
    title: str = "Top Features",
    palette: str = "viridis",
    filename: str | None = None,
) -> None:
    """Horizontal bar chart of the top-N features by importance."""
    setup_style()
    top = table.head(top_n)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top, x="Importance", y="Feature", palette=palette)
    plt.title(title)
    if filename:
        save_and_show(filename)
    else:
        plt.show()
        plt.close()


def plot_attack_comparison(df: pd.DataFrame, filename: str = "attack_comparison.png") -> None:
    """Bar chart comparing mean drift across attacks."""
    setup_style()
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="Attack", y="Drift", palette="Set2")
    plt.title("Attack Comparison - Explanation Drift")
    save_and_show(filename)


def plot_defense_comparison(df: pd.DataFrame, filename: str = "defense_comparison.png") -> None:
    """Bar chart comparing mean drift across defense models."""
    setup_style()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="Model", y="Drift")
    plt.title("Defense Architecture Comparison")
    save_and_show(filename)
