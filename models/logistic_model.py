"""
=====================================================================
XAI Attack and Defense Framework - Logistic Regression Model
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================

Thin wrapper around scikit-learn's `LogisticRegression` used as the linear
baseline classifier across all three cybersecurity datasets.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

import config


def build_logistic_model(**overrides) -> LogisticRegression:
    """Instantiate a Logistic Regression model with the project defaults."""
    kwargs = dict(config.LOGISTIC_CONFIG)
    kwargs.update(overrides)
    return LogisticRegression(**kwargs)


def train_logistic_model(X_train, y_train, **overrides) -> LogisticRegression:
    """Fit a Logistic Regression model on the provided training data."""
    model = build_logistic_model(**overrides)
    model.fit(X_train, y_train)
    return model
