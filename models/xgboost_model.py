"""
=====================================================================
XAI Attack and Defense Framework - XGBoost Model
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Wrapper around `xgboost.XGBClassifier` used as the primary tree-ensemble
model. XGBoost is chosen because it is highly accurate on tabular data and
is directly compatible with TreeSHAP for exact Shapley-value explanations.
"""

from __future__ import annotations

import xgboost as xgb

import config


def build_xgboost_model(**overrides) -> xgb.XGBClassifier:
    """Instantiate an XGBoost classifier with the project defaults."""
    kwargs = dict(config.XGBOOST_CONFIG)
    kwargs.update(overrides)
    return xgb.XGBClassifier(**kwargs)


def train_xgboost_model(X_train, y_train, **overrides) -> xgb.XGBClassifier:
    """Fit an XGBoost classifier on the provided training data."""
    model = build_xgboost_model(**overrides)
    model.fit(X_train, y_train)
    return model
