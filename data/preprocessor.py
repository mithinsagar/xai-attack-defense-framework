"""
=====================================================================
XAI Attack and Defense Framework - Data Preprocessor
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Implements the reusable preprocessing pipeline described in the
technical report Section 5.5:

    1. Drop non-informative or leakage columns.
    2. Factorise categorical columns (avoids one-hot dimensionality blow-up).
    3. Mean-impute missing values.
    4. StandardScaler normalisation.
    5. Stratified 80/20 train-test split.

Also persists processed arrays, feature name lists, and fitted scalers to
disk so downstream phases can load them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
from data.loader import load_dataset
from utils.io_utils import save_numpy, save_pickle
from utils.logger import get_logger

LOGGER = get_logger("data.preprocessor")


@dataclass
class PreprocessedDataset:
    key: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler


def preprocess_dataset(
    df: pd.DataFrame,
    label_column: str,
    drop_columns: Sequence[str] = (),
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_SEED,
    encode_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Preprocess a raw dataframe and return (X_train, X_test, y_train, y_test,
    feature_names, scaler).
    """
    LOGGER.info("Preprocessing dataset (label=%s)", label_column)

    df = df.drop(columns=list(drop_columns), errors="ignore")
    y = df[label_column]
    X = df.drop(columns=[label_column])

    feature_names = X.columns.tolist()

    # pandas 3 infers a native "str" dtype (or StringDtype/ArrowDtype in
    # other configurations) for plain string columns instead of the classic
    # "object" dtype this check used to assume, so anything non-numeric needs
    # factorising, not just columns literally typed "object".
    for col in X.columns:
        if not is_numeric_dtype(X[col]):
            X[col] = X[col].astype(str)
            X[col] = X[col].factorize()[0]

    X = X.fillna(X.mean(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if encode_labels or not is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    LOGGER.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test, feature_names, scaler


def preprocess_and_save(key: str) -> PreprocessedDataset:
    """Preprocess one dataset by key and persist arrays / scaler / features."""
    meta = config.DATASETS[key]
    df = load_dataset(key)

    X_train, X_test, y_train, y_test, features, scaler = preprocess_dataset(
        df=df,
        label_column=meta["label_column"],
        drop_columns=meta["drop_columns"],
        encode_labels=(key == "fraud"),
    )

    save_numpy(X_train, config.PROCESSED_DATA_DIR / f"X_train_{key}.npy")
    save_numpy(X_test, config.PROCESSED_DATA_DIR / f"X_test_{key}.npy")
    save_numpy(np.asarray(y_train), config.PROCESSED_DATA_DIR / f"y_train_{key}.npy")
    save_numpy(np.asarray(y_test), config.PROCESSED_DATA_DIR / f"y_test_{key}.npy")

    save_pickle(features, config.PROCESSED_DATA_DIR / f"feature_{key}.pkl")
    save_pickle(scaler, config.SCALERS_DIR / f"scaler_{key}.pkl")

    LOGGER.info("Saved processed artefacts for '%s'", key)
    return PreprocessedDataset(
        key=key,
        X_train=X_train,
        X_test=X_test,
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        feature_names=features,
        scaler=scaler,
    )


def preprocess_all_datasets() -> None:
    """Preprocess every configured dataset."""
    for key in config.DATASETS:
        preprocess_and_save(key)
