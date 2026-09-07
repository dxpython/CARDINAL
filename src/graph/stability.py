"""Edge selection frequency and direction agreement across resamples.

Given a collection of learned adjacency matrices (one per bootstrap resample or
per repeated CARDINAL run), compute per-edge selection frequency and the
direction-agreement rate relative to the prespecified reference structure.
"""

from __future__ import annotations

import numpy as np


def edge_frequencies(adjacencies: list[np.ndarray]) -> np.ndarray:
    """Selection frequency of each directed edge across fitted graphs."""
    stack = np.stack([np.asarray(a) > 0.5 for a in adjacencies]).astype(float)  # (R, d, d)
    return stack.mean(axis=0)


def direction_agreement(adjs: list[np.ndarray], reference: np.ndarray) -> float:
    """Fraction of reference edges whose learned direction matches."""

    def _dir(a):
        return np.asarray(a) > 0.5

    ref = _dir(reference)
    agree, total = 0.0, 0
    for a in adjs:
        aa = _dir(a)
        for i in range(ref.shape[0]):
            for j in range(ref.shape[1]):
                if ref[i, j]:
                    total += 1
                    if aa[i, j]:
                        agree += 1
    return agree / total if total else float("nan")


def stability_summary(adjacencies: list[np.ndarray], reference: np.ndarray,
                      feature_names: list[str]) -> dict:
    freq = edge_frequencies(adjacencies)
    da = direction_agreement(adjacencies, reference)
    return {"edge_frequency": freq, "direction_agreement": da,
            "feature_names": feature_names}
