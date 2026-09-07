"""Metric correctness on small, hand-checkable inputs."""

from __future__ import annotations

import numpy as np

from src.metrics.discrimination import discrimination_metrics
from src.metrics.calibration import brier_score, ece, nll_binary
from src.metrics.classification import confusion_counts, three_class_confusion
from src.statistics.delong import delong
from src.statistics.mcnemar import exact_mcnemar
from src.statistics.paired import paired_wilcoxon, holm_bonferroni, rank_biserial


def test_metric_known_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = discrimination_metrics(y, p, threshold=0.50)
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    # AUC of well-separated scores is 1.0
    assert m["auc"] == 1.0
    # confusion counts
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 2, 0, 0)


def test_brier_nll_ece():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.8, 0.7, 0.3])
    assert abs(brier_score(y, p) - np.mean((p - y) ** 2)) < 1e-9
    assert nll_binary(y, p) > 0
    assert 0.0 <= ece(y, p, n_bins=4) <= 1.0


def test_confusion_three_class():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 2])
    cm = three_class_confusion(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[1, 1] == 1 and cm[2, 1] == 1 and cm[1, 2] == 1


def test_delong_identical_scores_p_one():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    res = delong(y, s, s)
    assert res.pvalue > 0.99
    assert abs(res.delta_auc) < 1e-9


def test_mcnemar_expectation():
    y = np.array([0, 0, 1, 1])
    ref = np.array([0, 0, 1, 0])   # 1 error
    comp = np.array([0, 1, 1, 1])  # 1 error (different case)
    res = exact_mcnemar(y, ref, comp)
    assert res.b == 1 and res.c == 1
    assert res.pvalue == 1.0  # symmetric discordance


def test_wilcoxon_and_holm():
    x = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    res = paired_wilcoxon(x, y)
    assert res.median_delta > 0
    assert -1.0 <= res.rank_biserial <= 1.0
    adj = holm_bonferroni([0.01, 0.2, 0.3])
    assert all(a <= 1.0 for a in adj)
    assert adj[0] <= adj[1]
