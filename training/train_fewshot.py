"""
=====================================================================
XAI Attack and Defense Framework - Phase 5: Few-Shot Training
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Trains the Prototypical-Network few-shot encoder on the Phishing dataset
and evaluates its vulnerability under all four attacks. Writes
`fewshot_attack_results.csv` under `results/`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import config
from attacks.gradient_attack import GradientAttack
from attacks.prediction_stable_attack import PredictionStableAttack
from attacks.targeted_attack import TargetedAttack
from attacks.top_feature_attack import TopFeatureAttack
from explainers.integrated_gradients import compute_ig_attributions
from fewshot.fewshot_trainer import train_prototypical
from metrics.drift_metric import compute_all_drift_metrics
from utils.io_utils import load_numpy, load_pickle, save_torch_model
from utils.logger import get_logger

LOGGER = get_logger("training.fewshot")


def _run_all_attacks(model, sample: np.ndarray, features) -> List[Dict[str, Any]]:
    """Run every attack against the given model and one sample."""
    with torch.no_grad():
        target = int(
            torch.argmax(
                model(torch.tensor(sample.reshape(1, -1), dtype=torch.float32)), dim=1
            )
        )
    attr_before = compute_ig_attributions(model, sample, target=target)
    rows = []
    for atk in (
        TopFeatureAttack(features),
        PredictionStableAttack(features),
        GradientAttack(features),
        TargetedAttack(features),
    ):
        # Use IG top features for consistency
        order = np.argsort(attr_before)[::-1]
        top_features = [features[i] for i in order[: getattr(atk, "top_k", 3)]]
        try:
            perturbed = atk.perturb(sample, model, top_features=top_features)
        except TypeError:
            perturbed = atk.perturb(sample, model)
        attr_after = compute_ig_attributions(model, perturbed, target=target)
        rows.append({"Attack": atk.name, **compute_all_drift_metrics(attr_before, attr_after)})
    return rows


def train_and_evaluate_fewshot() -> pd.DataFrame:
    """Train the few-shot encoder and evaluate against all four attacks."""
    config.ensure_directories()
    try:
        X_train = load_numpy(config.PROCESSED_DATA_DIR / "X_train_phishing.npy")
        y_train = load_numpy(
            config.PROCESSED_DATA_DIR / "y_train_phishing.npy", allow_pickle=True
        ).astype(int)
        X_test = load_numpy(config.PROCESSED_DATA_DIR / "X_test_phishing.npy")
        features = load_pickle(config.PROCESSED_DATA_DIR / "feature_phishing.pkl")
    except FileNotFoundError as e:
        LOGGER.warning("Phishing tensors missing (%s). Skipping few-shot phase.", e)
        return pd.DataFrame()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)

    encoder = train_prototypical(Xt, yt)

    class _EncoderClassifier(torch.nn.Module):
        """Wrap encoder + fixed prototypes into a 2-class scorer for IG."""

        def __init__(self, encoder, prototypes):
            super().__init__()
            self.encoder = encoder
            self.prototypes = prototypes

        def forward(self, x):
            emb = self.encoder(x)
            distances = ((emb.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2).sum(2)
            return -distances

    # Build class-wise prototypes from training data
    with torch.no_grad():
        emb = encoder(Xt)
    prototypes = torch.stack(
        [emb[yt == c].mean(0) for c in torch.unique(yt)]
    )

    classifier = _EncoderClassifier(encoder, prototypes)
    save_torch_model(encoder, config.BASE_MODELS_DIR / "fewshot_encoder_phishing.pt")

    rows = _run_all_attacks(classifier, X_test[0], features)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "fewshot_attack_results.csv"
    df.to_csv(out, index=False)
    LOGGER.info("Few-shot attack results saved -> %s", out)
    return df
