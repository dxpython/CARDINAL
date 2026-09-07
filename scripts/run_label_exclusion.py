"""Reviewer-5 label-defining feature exclusion experiment.

Keeps the reference index Pd and normal/mild/severe labels identical, but removes
E/e', LAVI, and TR velocity from the predictor matrix and re-trains every model
from scratch under the same split / seed / preprocessing / protocol.  CARDINAL,
TabPFN-tuned, SAINT, and FT-Transformer are evaluated on the reduced-feature fold
and compared to the full-feature CARDINAL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (get_config, get_fold, metrics_frame, df_to_latex, save_fig,
                     setup_ax)
from src.experiments.common import run_model, SUMMARY_METRICS
from src.statistics.paired import paired_wilcoxon, holm_bonferroni
from src.utils import set_seed, save_csv


def main():
    cfg = get_config()
    seed = int(cfg.get("seed"))
    n_runs = cfg.get("num_repeated_runs", 20)

    # full-feature CARDINAL (reference)
    fold_full = get_fold(cfg, exclude_label_defining=False)
    res_card_full = run_model("CARDINAL", fold_full, cfg.as_dict(), seed=seed)

    # reduced-feature fold: E/e', LAVI, TR velocity removed from predictors
    fold_red = get_fold(cfg, exclude_label_defining=True)

    reduced_models = ["CARDINAL", "TabPFN-tuned", "SAINT", "FT-Transformer"]
    results = {}
    for m in reduced_models:
        results[m] = run_model(m, fold_red, cfg.as_dict(), seed=seed)

    # single-run metrics table (AUC/AUPRC/Acc/F1/Sens/Spec/Brier/ECE/NLL)
    rows = [{"model": "CARDINAL", "input_setting": "Full features",
             **{k: res_card_full.metrics.get(k, float("nan"))
                for k in SUMMARY_METRICS}}]
    for m in reduced_models:
        r = results[m]
        if r.error:
            rows.append({"model": m, "input_setting": "Excluding E/e', LAVI, TR velocity",
                         **{k: float("nan") for k in SUMMARY_METRICS}, "error": r.error})
        else:
            rows.append({"model": m, "input_setting": "Excluding E/e', LAVI, TR velocity",
                         **{k: r.metrics.get(k, float("nan")) for k in SUMMARY_METRICS}})
    perf = pd.DataFrame(rows)
    save_csv(perf, cfg.results_dir() / "label_feature_exclusion_results.csv")
    df_to_latex(perf, "Complementary label-exclusion analysis on the locked test "
                      "set.", "tab:labelexclusion", cfg, "labelexclusion")

    # delta vs full-feature CARDINAL
    if not res_card_full.error:
        full_auc = res_card_full.metrics.get("auc", float("nan"))
        deltas = []
        for m in reduced_models:
            r = results[m]
            if r.error:
                continue
            deltas.append({"model": m, "delta_auc_vs_full": r.metrics["auc"] - full_auc})
        if deltas:
            save_csv(pd.DataFrame(deltas), cfg.results_dir() / "label_exclusion_deltas.csv")

    # 20-run stability on reduced feature set
    aucs = {m: [] for m in reduced_models}
    for r_ in range(n_runs):
        s = seed + r_
        for m in list(aucs):
            res = run_model(m, fold_red, cfg.as_dict(), seed=s)
            if not res.error:
                aucs[m].append(res.metrics.get("auc", float("nan")))
    if all(len(v) == n_runs for v in aucs.values()):
        card = np.array(aucs["CARDINAL"])
        ps, rows = [], []
        for m in reduced_models:
            if m == "CARDINAL":
                continue
            pr = paired_wilcoxon(card, np.array(aucs[m]))
            rows.append({"model": m, "median_delta_auc": pr.median_delta,
                         "rank_biserial": pr.rank_biserial, "pvalue": pr.pvalue})
            ps.append(pr.pvalue)
        adj = holm_bonferroni(ps)
        for i, row in enumerate(rows):
            row["holm_p"] = adj[i]
        save_csv(pd.DataFrame(rows), cfg.results_dir() / "label_exclusion_repeated.csv")

    # comparison figure: full vs reduced feature set
    fig, ax = setup_ax(None, "Label-exclusion analysis", "Model", "AUC")
    models_plot = ["CARDINAL(full)", "CARDINAL", "TabPFN-tuned", "SAINT", "FT-Transformer"]
    auc_vals = [res_card_full.metrics.get("auc", np.nan)] + [
        results[m].metrics.get("auc", np.nan) for m in reduced_models]
    ax.bar(models_plot, auc_vals, color=["#1f77b4", "#2ca02c", "#7f7f7f", "#c5b0d5", "#ff7f0e"])
    ax.set_ylim(0.6, 1.0)
    save_fig(fig, "label_feature_exclusion_comparison", cfg)
    print("[run_label_exclusion] wrote results + figure")


if __name__ == "__main__":
    main()
