"""Aggregate result-figure generation (matplotlib, white background, academic).

Figures 1-3 are schematic diagrams drawn in PowerPoint and are NOT generated
here; this script produces the data-driven result figures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from _common import get_config, get_fold, save_fig, setup_ax
from src.experiments.common import train_cardinal, evaluate_cardinal
from src.models.cardinal import CARDINAL
from src.utils import set_seed


def baseline_distributions(fold, cfg):
    """Visualise the reference-index (Pd) distribution by reference-index grade."""
    grade2idx = {0: "Normal", 1: "Mild", 2: "Severe"}
    fig, ax = setup_ax(None, "Baseline distributions by grade", "", "Pd")
    for idx, (k, lbl) in enumerate(grade2idx.items()):
        vals = fold.pd_test[fold.y_test == k] if (fold.y_test == k).any() else np.array([np.nan])
        ax.plot([idx] * len(vals), vals, "o", ms=4, label=lbl)
    ax.set_xticks(list(grade2idx))
    ax.set_xticklabels(["Normal", "Mild", "Severe"])
    ax.legend()
    save_fig(fig, "02_baseline_distributions", cfg)


def confusion_matrix(fold, res, cfg):
    from src.metrics.classification import three_class_confusion
    cm = three_class_confusion(res.y_grade, res.grade_probabilities.argmax(axis=1))
    fig, ax = setup_ax(None, "Three-class confusion matrix", "Predicted", "Truth")
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Normal", "Mild", "Severe"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["Normal", "Mild", "Severe"])
    fig.colorbar(im)
    save_fig(fig, "14_confusion_matrix", cfg)


def learning_curves(res_list, cfg):
    fig, ax = setup_ax(None, "Training / validation loss", "Epoch", "Loss")
    for name, tr, va in res_list:
        ax.plot(tr, label=f"{name} train", alpha=0.7)
        ax.plot(va, ls="--", label=f"{name} val", alpha=0.7)
    ax.legend(fontsize=7)
    save_fig(fig, "11_learning_curves", cfg)


def feature_importance(fold, cfg, model):
    model_cfg = dict(cfg.get("model", {}))
    X = torch.tensor(fold.X_train, dtype=torch.float32)
    with torch.no_grad():
        A = torch.sigmoid(model.graph.edge_logits)
        z = model.encode(X, A)
    # proxy importance: |mean| latent derivative via graph contribution
    imp = (torch.mean(torch.abs(A), dim=0).numpy())
    fig, ax = setup_ax(None, "Global feature importance", "Feature", "|A| row-mean")
    names = fold.feature_names
    order = np.argsort(imp)[::-1][:12]
    ax.barh([names[i] for i in order][::-1], imp[order][::-1])
    save_fig(fig, "08_feature_importance", cfg)


def main():
    cfg = get_config()
    fold = get_fold(cfg)
    set_seed(int(cfg.get("seed")))
    model_cfg = dict(cfg.get("model", {}))
    model, tr, va = train_cardinal(fold, cfg.as_dict(), model_cfg)
    res = evaluate_cardinal(model, fold, cfg.as_dict())

    baseline_distributions(fold, cfg)
    confusion_matrix(fold, res, cfg)
    learning_curves([("CARDINAL", tr, va)], cfg)
    feature_importance(fold, cfg, model)
    print("[make_all_figures] wrote result figures")


if __name__ == "__main__":
    main()
