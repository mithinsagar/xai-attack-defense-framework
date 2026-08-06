"""
=====================================================================
XAI Attack and Defense Framework - Phase 2: Base Model Training
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Trains Logistic Regression, XGBoost, and MLP models on each of the three
cybersecurity datasets. Persists all models under `models/base_models/`
and writes an aggregated `baseline_results.csv` under `results/`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import config
from metrics.evaluation import evaluate_sklearn, evaluate_torch
from models.logistic_model import train_logistic_model
from models.mlp import MLP
from models.xgboost_model import train_xgboost_model
from utils.io_utils import load_numpy, save_pickle, save_torch_model
from utils.logger import get_logger

LOGGER = get_logger("training.base")


def _load_split(key: str):
    """Load processed train/test arrays for a dataset."""
    X_train = load_numpy(config.PROCESSED_DATA_DIR / f"X_train_{key}.npy")
    X_test = load_numpy(config.PROCESSED_DATA_DIR / f"X_test_{key}.npy")
    y_train = load_numpy(
        config.PROCESSED_DATA_DIR / f"y_train_{key}.npy", allow_pickle=True
    )
    y_test = load_numpy(
        config.PROCESSED_DATA_DIR / f"y_test_{key}.npy", allow_pickle=True
    )
    return X_train, X_test, y_train.astype(int), y_test.astype(int)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = config.MLP_TRAINING["epochs"],
    lr: float = config.MLP_TRAINING["learning_rate"],
    batch_size: int | None = None,
) -> nn.Module:
    input_dim = X_train.shape[1]
    model = MLP(input_dim)

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    if batch_size is None:
        # Full batch training (original notebook Phase 2)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                LOGGER.info("MLP | epoch %d | loss %.4f", epoch, loss.item())
    else:
        ds = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                total += loss.item()
            if epoch % 5 == 0:
                LOGGER.info("MLP | epoch %d | loss %.4f", epoch, total)
    return model


def train_all_base_models() -> pd.DataFrame:
    """Train all base models across all datasets and return summary DataFrame."""
    config.ensure_directories()
    rows: List[Dict[str, Any]] = []

    for key in config.DATASETS:
        LOGGER.info("=== Training base models for %s ===", key)
        try:
            X_train, X_test, y_train, y_test = _load_split(key)
        except FileNotFoundError as e:
            LOGGER.warning("Preprocessed data missing for %s: %s", key, e)
            continue

        # Logistic
        log = train_logistic_model(X_train, y_train)
        save_pickle(log, config.BASE_MODELS_DIR / f"logistic_{key}.pkl")
        m = evaluate_sklearn(log, X_test, y_test)
        rows.append({"Dataset": key, "Model": "Logistic", **m})

        # XGBoost
        xgbm = train_xgboost_model(X_train, y_train)
        save_pickle(xgbm, config.BASE_MODELS_DIR / f"xgb_{key}.pkl")
        m = evaluate_sklearn(xgbm, X_test, y_test)
        rows.append({"Dataset": key, "Model": "XGBoost", **m})

        # MLP  (IDS uses batched training)
        if key == "ids":
            mlp = _train_mlp(
                X_train, y_train,
                epochs=config.MLP_TRAINING["epochs_ids"],
                lr=config.MLP_TRAINING["learning_rate_ids"],
                batch_size=config.MLP_TRAINING["batch_size_ids"],
            )
        else:
            mlp = _train_mlp(X_train, y_train)
        save_torch_model(mlp, config.BASE_MODELS_DIR / f"nn_{key}.pt")
        m = evaluate_torch(mlp, X_test, y_test)
        rows.append({"Dataset": key, "Model": "MLP", **m})

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "baseline_results.csv"
    df.to_csv(out, index=False)
    LOGGER.info("Baseline results saved -> %s", out)
    return df
