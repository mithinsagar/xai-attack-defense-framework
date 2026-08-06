"""
=====================================================================
XAI Attack and Defense Framework - Figure Save Helpers
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import config


def save_and_show(name: str, dpi: int = 300) -> Path:
    """
    Save the current Matplotlib figure to `figures/<name>` and display it.
    Returns the resolved output path.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.show()
    plt.close()
    return path
