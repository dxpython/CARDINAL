"""Reference-concordance metrics: precision, recall, F1, structural Hamming distance.

These quantify agreement with the prespecified reference structure (the prior),
NOT causal truth.  All metrics are computed on directed adjacency matrices over
the same variable ordering.
"""

from __future__ import annotations

import numpy as np


def _binary(A: np.ndarray) -> np.ndarray:
    return (np.asarray(A) > 0.5).astype(int)


def structural_hamming_distance(A: np.ndarray, B: np.ndarray) -> int:
    """SHD: number of edge additions, deletions, and reversals."""
    a = _binary(A)
    b = _binary(B)
    shd = int(np.abs(a - b).sum())
    # reversal double-count correction
    rev = int(((a == 1) & (b == 1) & (b.T == 1) & (a.T == 1)).sum())
    return shd - rev


def precision_recall_f1(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    p = _binary(pred)
    r = _binary(ref)
    tp = int((p & r).sum())
    fp = int((p & ~r).sum())
    fn = int((~p & r).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def concordance(pred: np.ndarray, ref: np.ndarray) -> dict:
    out = precision_recall_f1(pred, ref)
    out["shd"] = structural_hamming_distance(pred, ref)
    return out
