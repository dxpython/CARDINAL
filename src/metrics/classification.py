"""Confusion-matrix helpers for both binary and three-class endpoints."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(tn), int(fp), int(fn)


def three_class_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                          labels=(0, 1, 2)) -> np.ndarray:
    """rows=truth, cols=prediction over normal/mild/severe grade codes."""
    return confusion_matrix(y_true, y_pred, labels=list(labels))
