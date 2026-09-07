"""Leak-controlled train / evaluate harness for CARDINAL and baselines.

All data-dependent operations (preprocessing, graph refinement, prototype
estimation, calibration, early stopping) are restricted to the training and
validation partitions; the locked test set is evaluated once at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.dataset import Fold
from ..metrics.calibration import ece, nll_binary
from ..metrics.discrimination import discrimination_metrics
from ..models.cardinal import CARDINAL
from ..models.graph import build_prior_from_edges
from ..models.baselines import get_baseline
from ..graph.prior import load_prior_edges
from ..utils import get_device, set_seed


@dataclass
class RunResult:
    model_name: str
    metrics: dict = field(default_factory=dict)          # binary endpoint metrics
    grader_metrics: dict = field(default_factory=dict)   # three-class confusion
    probs_abnormal: np.ndarray = field(default_factory=lambda: np.array([]))
    y_binary: np.ndarray = field(default_factory=lambda: np.array([]))
    y_grade: np.ndarray = field(default_factory=lambda: np.array([]))
    grade_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    predictive_variance: np.ndarray = field(default_factory=lambda: np.array([]))
    patient_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    loss_curve_train: list = field(default_factory=list)
    loss_curve_val: list = field(default_factory=list)
    trained_model: Optional[object] = None
    edge_frequency: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
    error: Optional[str] = None


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32))


def build_prior(fold: Fold, dataset: str = "clinical") -> np.ndarray:
    """Build A0 over the encoded feature columns from the prior edge list."""
    edges = load_prior_edges(dataset)
    return build_prior_from_edges(edges, fold.feature_names)


def train_cardinal(fold: Fold, cfg_common: dict, model_cfg: dict, seed: int = 20260730,
                   dataset: str = "clinical") -> tuple[CARDINAL, list, list]:
    device = get_device()
    set_seed(seed)

    Xtr = _to_tensor(fold.X_train).to(device)
    ytr = torch.tensor(fold.y_train, dtype=torch.long).to(device)
    Xva = _to_tensor(fold.X_val).to(device)
    yva = torch.tensor(fold.y_val, dtype=torch.long).to(device)

    num_classes = 3
    model = CARDINAL(fold.X_train.shape[1], num_classes, model_cfg).to(device)
    prior = build_prior(fold, dataset)
    model.set_prior(fold.X_train.shape[1], prior,
                    model_cfg.get("prior_weight", 0.25),
                    model_cfg.get("sparsity_weight", 0.05),
                    model_cfg.get("acyclicity_weight", 0.10))

    opt = torch.optim.AdamW(model.parameters(), lr=model_cfg.get("learning_rate", 1e-3),
                            weight_decay=model_cfg.get("weight_decay", 1e-4))
    max_epochs = model_cfg.get("max_epochs", 200)
    patience = model_cfg.get("early_stopping_patience", 20)
    batch_size = model_cfg.get("batch_size", 16)

    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    tr_loss, va_loss = [], []
    best_val, patience_ctr = float("inf"), 0
    initialized = False

    for epoch in range(max_epochs):
        model.train()
        ep = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            losses = model.compute_loss(xb, yb)
            losses["total"].backward()
            opt.step()
            ep += losses["total"].item() * len(xb)
        tr_loss.append(ep / len(Xtr))

        # prototype EMA update from (epoch-final) training representations
        model.eval()
        with torch.no_grad():
            A = model.learn_adjacency(use_gumbel=False)
            ztr = model.encode(Xtr, A)
            zva = model.encode(Xva, A)
        if not initialized:
            model.prototype.initialize_from(ztr, ytr)
            initialized = True
        else:
            model.prototype.update_ema(ztr, ytr)

        with torch.no_grad():
            logits_va = model.predict(zva)
            val_loss = nn.functional.cross_entropy(logits_va, yva)
        va_loss.append(val_loss.item())

        if val_loss.item() < best_val - 1e-6:
            best_val = val_loss.item()
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            break

    model.eval()
    return model, tr_loss, va_loss


def evaluate_cardinal(model: CARDINAL, fold: Fold, cfg: dict, seed: int = 20260730) -> RunResult:
    device = get_device()
    Xte = _to_tensor(fold.X_test).to(device)
    mc = cfg.get("uncertainty", {})
    T = mc.get("mc_passes_primary", 30)

    mean_prob, var_prob, entropy = model.mc_probabilities(Xte, T=T)
    mean_prob = mean_prob.cpu().numpy()
    var_prob = var_prob.cpu().numpy()
    entropy = entropy.cpu().numpy()

    # abnormal = mild(1) + severe(2); CARDINAL is trained on grades
    p_abnormal = (mean_prob[:, 1] + mean_prob[:, 2])
    y_bin = fold.yb_test
    y_grade = fold.y_test
    pred_grade = mean_prob.argmax(axis=1)

    metrics = discrimination_metrics(y_bin, p_abnormal, cfg["data"]["abnormal_threshold"])
    metrics["brier"] = float(np.mean((p_abnormal - y_bin) ** 2))
    metrics["ece"] = ece(y_bin, p_abnormal, mc.get("calibration_bins", 10))
    metrics["nll"] = nll_binary(y_bin, p_abnormal)

    from ..metrics.classification import three_class_confusion
    grad_cm = three_class_confusion(y_grade, pred_grade)

    return RunResult(
        model_name="CARDINAL", metrics=metrics,
        grader_metrics={"confusion": grad_cm,
                        "acc": float((pred_grade == y_grade).mean())},
        probs_abnormal=p_abnormal, y_binary=y_bin, y_grade=y_grade,
        grade_probabilities=mean_prob, uncertainty=entropy,
        predictive_variance=var_prob.sum(axis=1), patient_ids=fold.patient_ids,
        trained_model=model,
    )


def evaluate_baseline(name: str, fold: Fold, cfg: dict, seed: int = 20260730) -> RunResult:
    set_seed(seed)
    try:
        bl = get_baseline(name, n_features=fold.X_train.shape[1])
        bl.fit(fold.X_train, fold.yb_train)
        p_abnormal = bl.predict_proba_abnormal(fold.X_test)
    except (ImportError, Exception) as e:  # noqa: BLE001
        return RunResult(model_name=name, error=str(type(e).__name__) + ": " + str(e))

    y_bin = fold.yb_test
    metrics = discrimination_metrics(y_bin, p_abnormal, cfg["data"]["abnormal_threshold"])
    metrics["brier"] = float(np.mean((p_abnormal - y_bin) ** 2))
    metrics["ece"] = ece(y_bin, p_abnormal, cfg["uncertainty"].get("calibration_bins", 10))
    metrics["nll"] = nll_binary(y_bin, p_abnormal)
    return RunResult(model_name=name, metrics=metrics, probs_abnormal=p_abnormal,
                     y_binary=y_bin, patient_ids=fold.patient_ids)


SUMMARY_METRICS = ["auc", "auprc", "accuracy", "f1", "sensitivity", "specificity",
                   "brier", "ece", "nll"]


def run_model(model_name: str, fold: Fold, cfg: dict, seed: int = 20260730) -> RunResult:
    """Dispatch training + locked-test evaluation for CARDINAL or a baseline."""
    if model_name == "CARDINAL":
        model_cfg = dict(cfg.get("model", {}))
        model, tr, va = train_cardinal(fold, cfg, model_cfg, seed=seed)
        res = evaluate_cardinal(model, fold, cfg, seed=seed)
        res.loss_curve_train = tr
        res.loss_curve_val = va
        return res
    return evaluate_baseline(model_name, fold, cfg, seed=seed)
