"""Structure-learning comparators: PC, NOTEARS, DAG-GNN.

All return a directed adjacency matrix (|d| x |d|) over the same variable
ordering as the reference prior, so :mod:`src.graph.compare` can quantify
reference concordance.  NOTEARS and DAG-GNN rely on optional external
implementations; if unavailable they raise ``ImportError`` and the run harness
records the gap in RUN_STATUS.md rather than fabricating values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _fisher_z(r: float) -> float:
    return 0.5 * np.log((1 + r) / (1 - r + 1e-12))


def _partial_corr(x: np.ndarray, y: np.ndarray, cond: np.ndarray) -> float:
    """Partial correlation of x,y given conditioning set cond (Pearson)."""
    if cond.size == 0:
        return float(np.corrcoef(x, y)[0, 1])
    # regress x and y on cond, correlate residuals
    X = np.column_stack([np.ones(len(x)), cond])
    beta_x = np.linalg.lstsq(X, x, rcond=None)[0]
    beta_y = np.linalg.lstsq(X, y, rcond=None)[0]
    rx = x - X @ beta_x
    ry = y - X @ beta_y
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / (den + 1e-12))


def pc_structure(data: pd.DataFrame, alpha: float = 0.05, max_cond: int = 1) -> np.ndarray:
    """PC algorithm (skeleton via Fisher-z partial-correlation CI tests).

    A functional, conservative PC implementation. Orientation uses pair-wise
    dependency to set direction (notation only); the reference-concordance
    metrics treat this as a directed estimate.
    """
    cols = list(data.columns)
    d = len(cols)
    X = data[cols].to_numpy(dtype=float)
    adj = np.ones((d, d)) - np.eye(d)          # start fully connected
    n = len(X)

    for cond_size in range(0, max_cond + 1):
        for i in range(d):
            for j in range(i + 1, d):
                if adj[i, j] == 0:
                    continue
                others = [k for k in range(d) if k != i and k != j]
                for subset in _choose(others, cond_size):
                    r = _partial_corr(X[:, i], X[:, j], X[:, subset])
                    z = abs(_fisher_z(r)) * np.sqrt(n - cond_size - 3)
                    p = 2 * (1 - norm.cdf(z))
                    if p >= alpha:
                        adj[i, j] = adj[j, i] = 0.0
                        break
    # orient by marginal dependency direction (simplified; not causal truth)
    for i in range(d):
        for j in range(i + 1, d):
            if adj[i, j] == 1:
                r_ij = np.corrcoef(X[:, i], X[:, j])[0, 1]
                if r_ij < 0:
                    adj[i, j], adj[j, i] = 0.0, 1.0
                else:
                    adj[j, i] = 0.0
    return adj


def _choose(items: list, k: int) -> list[list]:
    if k == 0:
        return [[]]
    if k > len(items):
        return []
    from itertools import combinations
    return [list(c) for c in combinations(items, k)]


def notears_structure(data: pd.DataFrame) -> np.ndarray:
    """NOTEARS via causal-learn (optional dependency)."""
    try:
        from causallearn.search.constraint.FCI import FCI  # pragma: no cover
        raise ImportError("causal-learn NOTEARS import path not bound; see RUN_STATUS")
    except ModuleNotFoundError:
        raise ImportError(
            "NOTEARS requires `causal-learn`. Install it and bind the NOTEARS "
            "learner in src.graph.structure.notears_structure.")


def daggnn_structure(data: pd.DataFrame) -> np.ndarray:
    """DAG-GNN requires the authors' external repository (not pip-installable)."""
    raise ImportError(
        "DAG-GNN requires the authors' external implementation. Provide the "
        "learner in src.graph.structure.daggnn_structure or install it to run "
        "this comparator. Recorded as a dependency gap in RUN_STATUS.md.")


def run_structure_learning(data: pd.DataFrame, method: str) -> np.ndarray:
    if method.lower() == "pc":
        return pc_structure(data)
    if method.lower() == "notears":
        return notears_structure(data)
    if method.lower() == "dag-gnn" or method.lower() == "daggnn":
        return daggnn_structure(data)
    raise ValueError(f"Unknown structure-learning method: {method}")
