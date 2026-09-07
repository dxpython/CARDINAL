"""Uncertainty / calibration experiments: MC-dropout T sensitivity, reliability
diagram with Wilson CIs, and risk--coverage selective review."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _common import get_config, get_fold, save_fig, setup_ax
from src.experiments.common import train_cardinal, build_prior
from src.models.cardinal import CARDINAL
from src.metrics.discrimination import discrimination_metrics
from src.metrics.calibration import ece, brier_score, nll_binary, reliability_bins
from src.uncertainty.risk_coverage import risk_coverage, referral_summary
from src.utils import set_seed, save_csv


def main():
    cfg = get_config()
    seed = int(cfg.get("seed"))
    fold = get_fold(cfg)
    model_cfg = dict(cfg.get("model", {}))
    mc_cfg = cfg.get("uncertainty", {})

    set_seed(seed)
    model = CARDINAL(fold.X_train.shape[1], 3, model_cfg)
    device = "cpu"
    model.set_prior(fold.X_train.shape[1], build_prior(fold),
                    model_cfg.get("prior_weight", 0.25),
                    model_cfg.get("sparsity_weight", 0.05),
                    model_cfg.get("acyclicity_weight", 0.10))
    model, _, _ = train_cardinal(fold, cfg.as_dict(), model_cfg, seed=seed)

    Xte = torch.tensor(fold.X_test, dtype=torch.float32)
    yb = fold.yb_test

    # --- T sensitivity ---
    rows = []
    for T in mc_cfg.get("mc_passes_sensitivity", [5, 10, 20, 30, 50, 100]):
        mean_prob, var_prob, entropy = model.mc_probabilities(Xte, T=T)
        mean_prob = mean_prob.numpy()
        p_ab = mean_prob[:, 1] + mean_prob[:, 2]
        m = discrimination_metrics(yb, p_ab, cfg["data"]["abnormal_threshold"])
        rows.append({"T": T, "auc": m["auc"], "auprc": m["auprc"],
                     "accuracy": m["accuracy"], "f1": m["f1"],
                     "brier": brier_score(yb, p_ab),
                     "ece": ece(yb, p_ab, mc_cfg.get("calibration_bins", 10)),
                     "nll": nll_binary(yb, p_ab),
                     "mean_predictive_variance": float(var_prob.numpy().sum())})
    save_csv(pd.DataFrame(rows), cfg.results_dir() / "mc_dropout_sensitivity.csv")

    # --- primary T=30 reliability / risk-coverage ---
    mean_prob, var_prob, entropy = model.mc_probabilities(Xte, T=mc_cfg.get("mc_passes_primary", 30))
    mean_prob = mean_prob.numpy()
    p_ab = mean_prob[:, 1] + mean_prob[:, 2]
    entropy = entropy.numpy()

    rel = reliability_bins(yb, p_ab, mc_cfg.get("calibration_bins", 10))
    rel_df = pd.DataFrame(rel["bins"])
    save_csv(rel_df, cfg.results_dir() / "calibration_bins.csv")

    rc = risk_coverage(yb, p_ab, entropy)
    rc_df = pd.DataFrame({"coverage": rc["coverage"], "error": rc["error"]})
    save_csv(rc_df, cfg.results_dir() / "risk_coverage.csv")
    save_csv(pd.DataFrame(referral_summary(yb, p_ab, entropy)),
             cfg.results_dir() / "referral_summary.csv")

    # figures
    fig, ax = setup_ax(None, "MC dropout T sensitivity", "T", "AUC")
    ax.plot([r["T"] for r in rows], [r["auc"] for r in rows], "-o")
    save_fig(fig, "07_mc_dropout_sensitivity", cfg)

    fig, ax = setup_ax(None, "Reliability diagram", "Predicted probability", "Observed fraction")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.errorbar(rel_df["mean_predicted"], rel_df["observed_fraction"],
                yerr=[rel_df["mean_predicted"] - rel_df["observed_ci_low"],
                      rel_df["observed_ci_high"] - rel_df["mean_predicted"]],
                fmt="o", capsize=4)
    save_fig(fig, "04_calibration_curves", cfg)

    fig, ax = setup_ax(None, "Risk-coverage curve", "Coverage", "Classification error")
    ax.plot(rc_df["coverage"], rc_df["error"], "-o")
    save_fig(fig, "09_risk_coverage", cfg)
    print("[run_uncertainty] wrote uncertainty results + figures")


if __name__ == "__main__":
    main()
