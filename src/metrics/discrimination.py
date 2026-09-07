"""Discrimination and thresholded classification metrics (binary endpoint)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)

from .classification import confusion_counts


def discrimination_metrics(y_true: np.ndarray,
                           p_abnormal: np.ndarray,
                           threshold: float = 0.50) -> dict[str, float]:
    """Binary abnormal vs normal metrics at the paper's 0.50 threshold."""
    prob = np.asarray(p_abnormal, dtype=float)
    y = np.asarray(y_true, dtype=int)
    pred = (prob >= threshold).astype(int)

    np_pos = y.sum()
    out: dict[str, float] = {}
    out["auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    out["auprc"] = float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    out["accuracy"] = float(accuracy_score(y, pred))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    if np_pos > 0:
        out["sensitivity"] = float((pred[y == 1] == 1).mean())
    else:
        out["sensitivity"] = float("nan")
    if (y == 0).sum() > 0:
        out["specificity"] = float((pred[y == 0] == 0).mean())
    else:
        out["specificity"] = float("nan")
    tp, tn, fp, fn = confusion_counts(y, pred)
    out["tp"], out["tn"], out["fp"], out["fn"] = tp, tn, fp, fn
    return out
