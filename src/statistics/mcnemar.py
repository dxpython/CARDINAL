"""Exact McNemar test for paired binary classification errors.

b = cases reference-model wrong / comparator correct
c = cases reference-model correct / comparator wrong
Under H0, b ~ Binomial(b+c, 0.5).  Returns the exact two-sided p-value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


@dataclass
class McNemarResult:
    b: int
    c: int
    pvalue: float


def exact_mcnemar(y_true: np.ndarray, pred_ref: np.ndarray, pred_comp: np.ndarray) -> McNemarResult:
    y = np.asarray(y_true).astype(int)
    r = np.asarray(pred_ref).astype(int)
    c = np.asarray(pred_comp).astype(int)
    ref_correct = r == y
    comp_correct = c == y
    b = int((~ref_correct & comp_correct).sum())
    c_ = int((ref_correct & ~comp_correct).sum())
    total = b + c_
    if total == 0:
        p = 1.0
    else:
        p = float(binomtest(min(b, c_), total, p=0.5, alternative="two-sided").pvalue)
    return McNemarResult(b, c_, p)
