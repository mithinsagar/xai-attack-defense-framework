"""
=====================================================================
XAI Attack and Defense Framework - Top-Level Configuration
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Central configuration module. All hard-coded paths, hyperparameters, and
runtime constants used across the framework are defined here so that the
rest of the codebase does not need to hard-code any values.
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------
# Project root and directory layout
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(
    os.environ.get("XAI_ADF_ROOT", Path(__file__).resolve().parent)
)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = MODELS_DIR / "base_models"
DEFENSE_MODELS_DIR = MODELS_DIR / "defense_models"
SCALERS_DIR = MODELS_DIR / "scalers"

EXPLANATIONS_DIR = PROJECT_ROOT / "explanations"
ATTACKS_DIR = PROJECT_ROOT / "attacks_output"
METRICS_DIR = PROJECT_ROOT / "metrics_output"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

_ALL_DIRS = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    BASE_MODELS_DIR,
    DEFENSE_MODELS_DIR,
    SCALERS_DIR,
    EXPLANATIONS_DIR,
    ATTACKS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    LOGS_DIR,
]


def ensure_directories() -> None:
    """Create all project directories if they do not already exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Global runtime constants
# ---------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
BATCH_SIZE: int = 64
DEFAULT_EPOCHS: int = 30
LEARNING_RATE: float = 0.001


# ---------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------
DATASETS = {
    "phishing": {
        "filename": "PhiUSIIL_Phishing_URL_Dataset.csv",
        "label_column": "label",
        "drop_columns": [
            "FILENAME", "URL", "Domain", "Title", "URLSimilarityIndex",
        ],
        "class_names": ["Legitimate", "Phishing"],
        "source_url": (
            "https://archive-beta.ics.uci.edu/dataset/967/"
            "phiusiil+phishing+url+dataset"
        ),
    },
    "ids": {
        "filename": "cybersecurity_intrusion_data.csv",
        "label_column": "attack_detected",
        "drop_columns": ["session_id"],
        "class_names": ["Normal", "Attack"],
        "source_url": (
            "https://www.kaggle.com/datasets/dnkumars/"
            "cybersecurity-intrusion-detection-dataset"
        ),
    },
    "fraud": {
        "filename": "Fraudulent_online_shops_dataset.csv",
        "label_column": "Label",
        "drop_columns": ["Online shop URL", "Domain registration date"],
        "class_names": ["Legitimate", "Fraud"],
        "source_url": "https://data.mendeley.com/datasets/m7xtkx7g5m/1",
    },
}


# ---------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------
LOGISTIC_CONFIG = {
    "max_iter": 1000,
    "n_jobs": -1,
}

XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

MLP_CONFIG = {
    "hidden_layers": (256, 128, 64),
    "dropout": (0.3, 0.2, 0.0),
    "use_batchnorm": True,
    "output_dim": 1,
    "activation": "relu",
}

MLP_TRAINING = {
    "epochs": 20,
    "epochs_ids": 40,
    "batch_size_ids": 32,
    "learning_rate": 0.001,
    "learning_rate_ids": 0.0005,
}


# ---------------------------------------------------------------------
# Attack hyperparameters
# ---------------------------------------------------------------------
ATTACK_CONFIG = {
    "attack1_epsilon": 0.05,
    "attack1_top_k": 3,
    "attack2_epsilon": 0.10,
    "attack2_low_k": 3,
    "attack3_epsilon": 0.05,
    "attack3_top_k": 3,
    "attack4_epsilon": 0.25,
    "attack4_top_k": 5,
}


# ---------------------------------------------------------------------
# Defense hyperparameters
# ---------------------------------------------------------------------
DEFENSE_CONFIG = {
    "defense1": {
        "epochs": 50,
        "noise_sigma": 0.01,
        "stability_weight": 0.1,
        "lr": 0.001,
    },
    "defense2": {
        "epochs": 60,
        "noise_sigma": 0.05,
        "adv_weight": 0.5,
        "lr": 0.001,
    },
    "defense3": {
        "epochs": 40,
        "noise_sigma": 0.02,
        "explain_weight": 0.2,
        "ig_batch_size": 200,
        "lr": 0.001,
    },
    "defense4": {
        "epochs": 50,
        "small_noise": 0.01,
        "large_noise": 0.05,
        "stability_weight": 0.2,
        "adv_weight": 0.2,
        "smooth_weight": 0.1,
        "lr": 0.001,
    },
}


# ---------------------------------------------------------------------
# Few-shot learning
# ---------------------------------------------------------------------
FEWSHOT_CONFIG = {
    "samples_per_class": 50,
    "n_way": 2,
    "k_shot": 5,
    "q_query": 10,
    "encoder_hidden": 128,
    "encoder_output": 64,
    "epochs": 300,
    "lr": 0.001,
}


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------
AUTHOR = "Mithin Sagar S"
GITHUB_URL = "https://github.com/mithinsagar"
PROJECT_NAME = "XAI Attack and Defense Framework"


if __name__ == "__main__":
    ensure_directories()
    print(f"[{PROJECT_NAME}] project directories initialised.")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
