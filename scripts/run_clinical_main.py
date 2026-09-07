"""Clinical cohort main experiment (paper Table 1 + repeated-run stability).

Runs every comparator and CARDINAL on the locked test set, then repeated matched
runs and patient-level DeLong / McNemar comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (get_config, get_schema, get_fold, metrics_frame,
                     df_to_latex, run_models)
from src.experiments.common import run_model
from src.experiments.common import SUMMARY_METRICS
from src.metrics.discrimination import discrimination_metrics
from src.statistics.delong import delong
from src.statistics.mcnemar import exact_mcnemar
from src.statistics.paired import paired_wilcoxon, holm_bonferroni
from src.utils import set_seed, save_csv


def main():
    cfg = get_config()
    fold = get_fold(cfg)
    models = cfg.get("baselines", {}).get("enabled", [])
    if "CARDINAL" not in models:
        models = ["CARDINAL"] + list(models)

    # --- locked test set: single-run metrics per model ---
    results = {}
    for m in models:
        results[m] = run_model(m, fold, cfg.as_dict(), seed=int(cfg.get("seed")))
    perf = metrics_frame(results)
    perf.insert(1, "split", "test")
    perf.insert(2, "n", len(fold.yb_test))
    save_csv(perf, cfg.results_dir() / "model_performance_test.csv")
    df_to_latex(perf[["model"] + SUMMARY_METRICS], "Binary abnormal-versus-normal "
                "discrimination, classification, and calibration on the locked "
                "18-case test set.", "tab:testperformance", cfg, "testperformance")

    # --- patient-level DeLong + McNemar (CARDINAL vs each strong baseline) ---
    card = results.get("CARDINAL")
    comp_names = [m for m in ["TabPFN", "TabPFN-tuned", "SAINT", "FT-Transformer"] if m in results]
    paired_rows = []
    for name in comp_names:
        res = results[name]
        if res.error or card.error:
            continue
        d = delong(fold.yb_test, card.probs_abnormal, res.probs_abnormal)
        pref = (card.probs_abnormal >= 0.50).astype(int)
        crei = (res.probs_abnormal >= 0.50).astype(int)
        mcn = exact_mcnemar(fold.yb_test, pref, crei)
        paired_rows.append({
            "reference_model": "CARDINAL", "comparator_model": name, "split": "test",
            "n": len(fold.yb_test), "reference_auc": d.auc1, "comparator_auc": d.auc2,
            "delta_auc": d.delta_auc, "delong_ci_low": d.ci_low, "delong_ci_high": d.ci_high,
            "delong_p": d.pvalue, "mcnemar_p": mcn.pvalue})
    paired_df = pd.DataFrame(paired_rows)
    if not paired_df.empty:
        save_csv(paired_df, cfg.results_dir() / "paired_model_tests.csv")
        df_to_latex(paired_df[["comparator_model", "delta_auc", "delong_ci_low",
                               "delong_ci_high", "delong_p", "mcnemar_p"]],
                    "Patient-level paired statistical comparisons on the fixed "
                    "locked test set.", "tab:paired", cfg, "paired")

    # --- repeated matched runs (20 seeds) for stability ---
    n_runs = cfg.get("num_repeated_runs", 20)
    stability_models = ["CARDINAL", "TabPFN-tuned", "SAINT", "FT-Transformer"]
    aucs = {m: [] for m in stability_models if m in models}
    seed_base = int(cfg.get("seed"))
    for r in range(n_runs):
        seed = seed_base + r
        for m in list(aucs):
            res = run_model(m, fold, cfg.as_dict(), seed=seed)
            if res.error:
                continue
            aucs[m].append(res.metrics.get("auc", float("nan")))

    rows = []
    for m, vals in aucs.items():
        for k, v in enumerate(vals):
            rows.append({"model": m, "run_seed": seed_base + k, "auc": v,
                         "simulation_seed": seed_base})
    rep = pd.DataFrame(rows)
    save_csv(rep, cfg.results_dir() / "repeated_seed_tests.csv")

    # paired Wilcoxon on matched runs + Holm
    if "CARDINAL" in aucs and len(aucs["CARDINAL"]) > 1:
        card_aucs = np.array(aucs["CARDINAL"])
        rows = []
        ps = []
        comp_list = [(m, np.array(aucs[m])) for m in aucs if m != "CARDINAL"
                     and len(aucs[m]) == len(card_aucs)]
        for m, v in comp_list:
            pr = paired_wilcoxon(card_aucs, v)
            rows.append({"dataset": "Clinical cohort", "comparator": m,
                         "median_delta_auc": pr.median_delta,
                         "rank_biserial": pr.rank_biserial, "pvalue": pr.pvalue})
            ps.append(pr.pvalue)
        if rows:
            adj = holm_bonferroni(ps)
            for i, row in enumerate(rows):
                row["holm_p"] = adj[i]
            save_csv(pd.DataFrame(rows), cfg.results_dir() / "repeated_run_paired.csv")

    print("[run_clinical_main] wrote result CSVs to", cfg.results_dir())


if __name__ == "__main__":
    main()
