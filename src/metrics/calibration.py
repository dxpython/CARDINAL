"""Probabilistic calibration metrics: Brier, ECE, NLL, reliability bins + Wilson CI."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Binary Brier score: mean((p - y)^2)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(((p - y) ** 2).mean())


def nll_binary(y_true: np.ndarray, p_abnormal: np.ndarray) -> float:
    """Negative log-likelihood for the binary endpoint (clipped for stability)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_abnormal, dtype=float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10,
        quantile: bool = True) -> float:
    """Expected calibration error over bins of predicted probability."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    bins, bin_edges = _bins(p, n_bins, quantile)
    total = 0.0
    for b in np.unique(bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        total += (mask.sum() / len(y)) * abs(acc - conf)
    return float(total)


def reliability_bins(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10,
                    confidence: float = 0.95, quantile: bool = True) -> dict:
    """Per-bin reliability summary with Wilson 95% CI for observed frequency."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    bins, edges = _bins(p, n_bins, quantile)
    recs = []
    for b in sorted(np.unique(bins)):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        obs = float(y[mask].mean())
        lo, hi = wilson_interval(int(y[mask].sum()), n, confidence)
        recs.append({"bin": int(b), "bin_lower": float(p[mask].min()),
                     "bin_upper": float(p[mask].max()),
                     "mean_predicted": float(p[mask].mean()),
                     "observed_fraction": obs,
                     "observed_ci_low": lo, "observed_ci_high": hi, "n": n})
    return {"bins": recs, "edges": edges.tolist()}


def _bins(p: np.ndarray, n_bins: int, quantile: bool = True):
    if quantile:
        try:
            # unique quantile edges to avoid empty bins
            edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
            bins = np.digitize(p, edges, right=True).clip(0, len(edges) - 1)
            return bins, edges
        except Exception:
            pass
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.digitize(p, edges, right=False) - 1
    return bins, edges


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a Binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = -ndtri((1 - confidence) / 2.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = max(0.0, (centre - margin) / denom)
    hi = min(1.0, (centre + margin) / denom)
    return float(lo), float(hi)
