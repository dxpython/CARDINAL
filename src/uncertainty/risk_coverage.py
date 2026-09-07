"""Risk--coverage analysis for uncertainty-guided selective review.

Cases are ranked by predictive uncertainty (entropy or predictive variance);
deferring the most uncertain cases, classification error on retained cases is
measured as a function of coverage.
"""

from __future__ import annotations

import numpy as np


def risk_coverage(y_true: np.ndarray, p_abnormal: np.ndarray, uncertainty: np.ndarray) -> dict:
    """Evaluate error vs coverage when cases are ranked by ``uncertainty``.

    Returns arrays over decreasing coverage (most-certain subset first).
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_abnormal, dtype=float)
    u = np.asarray(uncertainty, dtype=float)
    order = np.argsort(u)           # ascending uncertainty
    y_s = y[order]
    p_s = p[order]
    n = len(y_s)

    coverages = []
    errors = []
    for k in range(1, n + 1):
        pred = (p_s[:k] >= 0.50).astype(int)
        err = float((pred != y_s[:k]).mean()) if k > 0 else float("nan")
        coverages.append(float(k / n))
        errors.append(err)
    return {"coverage": np.array(coverages), "error": np.array(errors)}


def referral_summary(y_true: np.ndarray, p_abnormal: np.ndarray, uncertainty: np.ndarray,
                     coverage_targets=(0.80, 0.90)) -> list[dict]:
    """Sensitivity of retained-case error at selected coverage levels."""
    rc = risk_coverage(y_true, p_abnormal, uncertainty)
    out = []
    cov, err = rc["coverage"], rc["error"]
    for target in coverage_targets:
        idx = int(np.abs(cov - target).argmin())
        out.append({"coverage": float(cov[idx]), "retained_error": float(err[idx])})
    return out
