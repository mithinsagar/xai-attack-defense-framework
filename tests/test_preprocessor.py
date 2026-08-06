"""
=====================================================================
XAI Attack and Defense Framework - Tests: Preprocessor
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.preprocessor import preprocess_dataset


def _fake_dataframe(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "cat": rng.choice(["a", "b", "c"], size=n),
        "label": rng.integers(0, 2, size=n),
    })


def test_preprocess_shapes_and_scaling():
    df = _fake_dataframe()
    X_train, X_test, y_train, y_test, features, scaler = preprocess_dataset(
        df=df, label_column="label", drop_columns=(),
    )

    assert X_train.shape[1] == 3  # 3 feature columns (f1, f2, cat)
    assert len(features) == 3
    assert X_train.shape[0] + X_test.shape[0] == len(df)
    assert len(y_train) + len(y_test) == len(df)

    # StandardScaler should give roughly zero mean per column
    assert np.abs(X_train.mean(axis=0)).max() < 1.0


def test_preprocess_stratified_split_preserves_classes():
    df = _fake_dataframe()
    _, _, y_train, y_test, _, _ = preprocess_dataset(
        df=df, label_column="label", drop_columns=(),
    )
    assert set(np.unique(y_train)) == set(np.unique(y_test))
