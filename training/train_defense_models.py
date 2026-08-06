"""
=====================================================================
XAI Attack and Defense Framework - Phase 6: Defense Training
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Trains all four defense models on the Phishing dataset (following the
original report), evaluates their explanation drift under a canonical
Attack-1 perturbation, and writes `defense_results.csv`.

The best-performing model (Defense 4) is saved to:
    models/defense_models/best_defense_model.pt
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import config
from attacks.top_feature_attack import TopFeatureAttack
from defenses.adversarial_training import train_adversarial
from defenses.explanation_regularization import train_explanation_reg
from defenses.hybrid_defense import train_hybrid
from defenses.stability_training import train_stability
from explainers.integrated_gradients import compute_ig_attributions
from metrics.drift_metric import compute_all_drift_metrics
from utils.io_utils import load_numpy, load_pickle, save_torch_model
from utils.logger import get_logger

LOGGER = get_logger("training.defense")


def _load_phishing_tensors():
    X_train = load_numpy(config.PROCESSED_DATA_DIR / "X_train_phishing.npy")
    y_train = load_numpy(
        config.PROCESSED_DATA_DIR / "y_train_phishing.npy", allow_pickle=True
    ).astype(int)
    X_test = load_numpy(config.PROCESSED_DATA_DIR / "X_test_phishing.npy")

    Xtr, Xv, ytr, yv = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y_train
    )
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xv, dtype=torch.float32)
    yv = torch.tensor(yv, dtype=torch.long)
    return Xtr, ytr, Xv, yv, X_test


def _evaluate_defense(model: torch.nn.Module, X_test: np.ndarray, features) -> Dict[str, float]:
    """Evaluate a defense model against Attack 1 (top-feature perturbation)."""
    sample = X_test[0]
    model.eval()

    with torch.no_grad():
        target = int(
            torch.argmax(model(torch.tensor(sample.reshape(1, -1), dtype=torch.float32)), dim=1)
        )

    attack = TopFeatureAttack(features)
    attr_before = compute_ig_attributions(model, sample, target=target)
    # Use the pre-computed IG top features
    order = np.argsort(attr_before)[::-1]
    top_features = [features[i] for i in order[: attack.top_k]]
    perturbed = attack.perturb(sample, model, top_features=top_features)
    attr_after = compute_ig_attributions(model, perturbed, target=target)
    return compute_all_drift_metrics(attr_before, attr_after)


def train_all_defenses() -> pd.DataFrame:
    config.ensure_directories()
    try:
        Xtr, ytr, Xv, yv, X_test = _load_phishing_tensors()
    except FileNotFoundError as e:
        LOGGER.warning("Phishing tensors not available (%s). Skipping.", e)
        return pd.DataFrame()

    features = load_pickle(config.PROCESSED_DATA_DIR / "feature_phishing.pkl")

    rows: List[Dict[str, Any]] = []

    LOGGER.info("Training Defense 1: Stability")
    d1 = train_stability(Xtr, ytr, Xv, yv)
    save_torch_model(d1, config.DEFENSE_MODELS_DIR / "defense1_stability.pt")
    rows.append({"Defense": "defense1_stability", **_evaluate_defense(d1, X_test, features)})

    LOGGER.info("Training Defense 2: Adversarial")
    d2 = train_adversarial(Xtr, ytr, Xv, yv)
    save_torch_model(d2, config.DEFENSE_MODELS_DIR / "defense2_adversarial.pt")
    rows.append({"Defense": "defense2_adversarial", **_evaluate_defense(d2, X_test, features)})

    LOGGER.info("Training Defense 3: Explanation Regularisation")
    d3 = train_explanation_reg(Xtr, ytr)
    save_torch_model(d3, config.DEFENSE_MODELS_DIR / "defense3_explanation_reg.pt")
    rows.append({"Defense": "defense3_explanation_reg", **_evaluate_defense(d3, X_test, features)})

    LOGGER.info("Training Defense 4: Hybrid (BEST)")
    d4 = train_hybrid(Xtr, ytr, Xv, yv)
    save_torch_model(d4, config.DEFENSE_MODELS_DIR / "defense4_hybrid.pt")
    save_torch_model(d4, config.DEFENSE_MODELS_DIR / "best_defense_model.pt")
    rows.append({"Defense": "defense4_hybrid", **_evaluate_defense(d4, X_test, features)})

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "defense_results.csv"
    df.to_csv(out, index=False)
    LOGGER.info("Defense results saved -> %s", out)
    return df
