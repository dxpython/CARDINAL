"""Component ablation: Full CARDINAL vs removing graph / prototype / calibration.

Ablations are implemented by zeroing the corresponding loss weight in the
objective  L = L_CE + alpha*L_graph + beta*L_proto + gamma*L_cal.  All ablation
variants and seeds use the identical leakage-controlled protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import get_config, get_fold, save_fig, setup_ax
from src.experiments.common import train_cardinal, evaluate_cardinal
from src.utils import set_seed, save_csv

ABLATIONS = {
    "Full CARDINAL": {"alpha_graph": 1.0, "beta_proto": 1.0, "gamma_cal": 0.10},
    "Without graph": {"alpha_graph": 0.0, "beta_proto": 1.0, "gamma_cal": 0.10},
    "Without prototype": {"alpha_graph": 1.0, "beta_proto": 0.0, "gamma_cal": 0.10},
    "Without calibration": {"alpha_graph": 1.0, "beta_proto": 1.0, "gamma_cal": 0.0},
}


def main():
    cfg = get_config()
    fold = get_fold(cfg)
    seed = int(cfg.get("seed"))
    n_runs = 20
    base_cfg = dict(cfg.get("model", {}))

    rows = []
    per_seed = {name: {"seed": [], "auc": []} for name in ABLATIONS}
    for name, weights in ABLATIONS.items():
        model_cfg = dict(base_cfg)
        model_cfg.update(weights)
        aucs = []
        for r in range(n_runs):
            set_seed(seed + r)
            m, tr, va = train_cardinal(fold, cfg.as_dict(), model_cfg, seed=seed + r)
            res = evaluate_cardinal(m, fold, cfg.as_dict(), seed=seed + r)
            auc = res.metrics.get("auc", float("nan"))
            aucs.append(auc)
            per_seed[name]["seed"].append(seed + r)
            per_seed[name]["auc"].append(auc)
        rows.append({"variant": name, "mean_auc": float(np.nanmean(aucs)),
                     "sd_auc": float(np.nanstd(aucs)),
                     "ci_low": float(np.nanmean(aucs) - 1.96 * np.nanstd(aucs)),
                     "ci_high": float(np.nanmean(aucs) + 1.96 * np.nanstd(aucs))})

    # delta relative to full model
    full_mean = next(r["mean_auc"] for r in rows if r["variant"] == "Full CARDINAL")
    for r in rows:
        r["delta_vs_full"] = r["mean_auc"] - full_mean
    abl = pd.DataFrame(rows)
    save_csv(abl, cfg.results_dir() / "ablation_results.csv")
    save_csv(pd.DataFrame({k: v for k, v in per_seed.items()}).T.reset_index(),
             cfg.results_dir() / "ablation_per_seed.csv")

    # forest plot
    fig, ax = setup_ax(None, "Component ablation", "Mean AUC", "Variant")
    y = np.arange(len(abl))[::-1]
    ax.errorbar(abl["mean_auc"], y, xerr=1.96 * abl["sd_auc"], fmt="o",
                color="black", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(abl["variant"])
    ax.axvline(full_mean, ls="--", color="tab:blue")
    save_fig(fig, "04_ablation", cfg)
    print("[run_ablation] wrote ablation results + figure")


if __name__ == "__main__":
    main()
