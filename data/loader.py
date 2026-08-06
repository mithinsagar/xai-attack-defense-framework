"""
=====================================================================
XAI Attack and Defense Framework - Dataset Loader
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Loads raw CSVs for the three cybersecurity datasets used in this project.

Expected directory layout:
    data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
    data/raw/cybersecurity_intrusion_data.csv
    data/raw/Fraudulent_online_shops_dataset.csv

Sources:
    PhiUSIIL Phishing URL Dataset:
        https://archive-beta.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
    Cybersecurity Intrusion Detection:
        https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset
    Fraudulent Online Shops:
        https://data.mendeley.com/datasets/m7xtkx7g5m/1
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

import config
from utils.logger import get_logger

LOGGER = get_logger("data.loader")


def load_dataset(key: str) -> pd.DataFrame:
    """Load one of the configured datasets by its short key."""
    if key not in config.DATASETS:
        raise KeyError(
            f"Unknown dataset key '{key}'. "
            f"Expected one of {list(config.DATASETS)}."
        )
    meta = config.DATASETS[key]
    path = config.RAW_DATA_DIR / meta["filename"]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            f"Download it from {meta['source_url']} and place it in "
            f"'{config.RAW_DATA_DIR}'."
        )
    LOGGER.info("Loading %s dataset from %s", key, path)
    df = pd.read_csv(path)
    LOGGER.info("%s dataset shape: %s", key, df.shape)
    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """Load all three datasets and return as a dictionary keyed by short key."""
    return {key: load_dataset(key) for key in config.DATASETS}
