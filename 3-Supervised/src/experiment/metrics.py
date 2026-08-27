"""
Metric helpers for the supervised-learning experiment.

This module detects the minority class from the training labels and exposes both a direct
scoring function and a scikit-learn compatible scorer.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer


def detect_minority_class(y: pd.Series | np.ndarray) -> Any:
    """
    Return the least frequent label in a target vector.

    Args:
        y (pd.Series | np.ndarray): Target labels.

    Returns:
        Any: The minority class label.
    """
    series = pd.Series(np.asarray(y).ravel())
    counts = series.value_counts()
    return counts.index[-1]


def resolve_minority_class(y: pd.Series | np.ndarray, configured: str | None) -> Any:
    """
    Resolve the minority class from configuration or by detection.

    Args:
        y (pd.Series | np.ndarray): Target labels used for auto-detection.
        configured (str | None): Raw configuration value. ``"auto"`` (or an
            empty value) triggers detection; anything else is parsed as the
            explicit label.

    Returns:
        Any: The resolved minority class label.
    """
    if configured is None or str(configured).strip().lower() in {"", "auto"}:
        return detect_minority_class(y)

    raw = str(configured).strip()
    labels = pd.Series(np.asarray(y).ravel()).unique()
    for label in labels:
        if str(label) == raw:
            return label
    # Fall back to numeric parsing when the label types differ from strings.
    try:
        return type(labels[0])(raw)
    except (ValueError, TypeError):
        return raw


def f1_minority_score(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    minority_class: Any,
) -> float:
    """
    Compute the F1 score for the minority class only.

    Args:
        y_true (pd.Series | np.ndarray): Ground-truth labels.
        y_pred (pd.Series | np.ndarray): Predicted labels.
        minority_class (Any): Label treated as the positive (minority) class.

    Returns:
        float: F1 score for the minority class.
    """
    return float(f1_score(y_true, y_pred, pos_label=minority_class, zero_division=0))


def make_f1_minority_scorer(minority_class: Any) -> Any:
    """
    Build a scikit-learn scorer for the minority-class F1.

    Args:
        minority_class (Any): Label treated as the positive (minority) class.

    Returns:
        Any: A scorer callable compatible with scikit-learn estimators.
    """
    return make_scorer(
        f1_score,
        pos_label=minority_class,
        zero_division=0,
    )
