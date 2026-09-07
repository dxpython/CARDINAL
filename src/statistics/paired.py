"""Repeated-run paired tests: Wilcoxon signed-rank, matched-pairs rank-biserial
effect size, and Holm-Bonferroni adjustment across comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class PairedResult:
    median_delta: float
    w: float
    pvalue: float
    rank_biserial: float


def rank_biserial(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014 form).

    r = (W+ - W-) / (W+ + W-), using signed ranks of the absolute differences.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return float("nan")
    ranks = np.abs(d).argsort().argsort() + 1  # rank of |d|
    signed = np.sign(d) * ranks
    w_plus = signed[signed > 0].sum()
    w_minus = -signed[signed < 0].sum()
    denom = w_plus + w_minus
    return float((w_plus - w_minus) / denom) if denom > 0 else float("nan")


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> PairedResult:
    """Wilcoxon signed-rank test on matched per-run metrics with effect size."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y
    try:
        stat, p = wilcoxon(x, y)
    except ValueError:
        # all differences zero -> no signal
        stat, p = 0.0, 1.0
    return PairedResult(median_delta=float(np.median(diffs)),
                        w=float(stat), pvalue=float(p),
                        rank_biserial=rank_biserial(diffs))


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (input order preserved)."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank, idx in enumerate(order):
        v = min(1.0, max(running, (n - rank) * p[idx]))
        adj[idx] = v
        running = v
    return adj.tolist()
