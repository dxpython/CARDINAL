"""DeLong non-parametric comparison of two correlated ROC-AUCs.

Implements the DeLong et al. covariance estimator as the two-sample
Mann-Whitney U-statistic kernel. Returns AUCs, variances, covariance, z, p and a
two-sided 95% CI on the AUC difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class DeLongResult:
    auc1: float
    auc2: float
    delta_auc: float
    var1: float
    var2: float
    cov: float
    z: float
    pvalue: float
    ci_low: float
    ci_high: float


def _structural(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (V10 over positives, V01 over negatives, AUC)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    sp = s[y]
    sn = s[~y]
    n_pos = len(sp)
    n_neg = len(sn)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes are required for DeLong.")
    # V10_i = mean over negatives of I(sp_i > sn_j)
    v10 = (sp[:, None] > sn[None, :]).mean(axis=1)
    # V01_j = mean over positives of I(sp_i > sn_j)
    v01 = (sn[None, :] < sp[:, None]).mean(axis=0)
    return v10, v01, float(v10.mean())


def _sample_var(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.var(x, ddof=1))


def delong(y: np.ndarray, s1: np.ndarray, s2: np.ndarray,
           alpha: float = 0.05) -> DeLongResult:
    y = np.asarray(y).astype(int)
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    v10_1, v01_1, auc1 = _structural(y, s1)
    v10_2, v01_2, auc2 = _structural(y, s2)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    var1 = _sample_var(v10_1) / n_pos + _sample_var(v01_1) / n_neg
    var2 = _sample_var(v10_2) / n_pos + _sample_var(v01_2) / n_neg
    cov = (np.cov(v10_1, v10_2, ddof=1)[0, 1] / n_pos
           + np.cov(v01_1, v01_2, ddof=1)[0, 1] / n_neg)

    se = np.sqrt(max(0.0, var1 + var2 - 2 * cov))
    z = (auc1 - auc2) / se if se > 0 else 0.0
    pvalue = float(2 * (1 - norm.cdf(abs(z))))
    zc = norm.ppf(1 - alpha / 2)
    ci_low = float((auc1 - auc2) - zc * se)
    ci_high = float((auc1 - auc2) + zc * se)
    return DeLongResult(auc1, auc2, auc1 - auc2, var1, var2, float(cov),
                        float(z), pvalue, ci_low, ci_high)
