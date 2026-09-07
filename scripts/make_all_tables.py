"""Render LaTeX tables for the main and supplementary results from saved CSVs."""

from __future__ import annotations

import pandas as pd

from _common import get_config, df_to_latex

TABLES = [
    ("model_performance_test", "Binary abnormal-versus-normal discrimination, "
     "classification, and calibration on the locked test set.", "tab:testperformance"),
    ("paired_model_tests", "Patient-level paired statistical comparisons on the "
     "locked test set.", "tab:paired"),
    ("label_feature_exclusion_results", "Complementary label-exclusion analysis "
     "on the locked test set.", "tab:labelexclusion"),
    ("mc_dropout_sensitivity", "Sensitivity of metrics to the number of "
     "stochastic forward passes T.", "tab:s_mc"),
    ("graph_stability", "Structural agreement with the prespecified reference "
     "structure.", "tab:s_graph"),
    ("benchmark_summary", "Public tabular robustness benchmarks.", "tab:benchmarks"),
]


def main():
    cfg = get_config()
    for name, caption, label in TABLES:
        csv = cfg.results_dir() / f"{name}.csv"
        if not csv.exists():
            print(f"  skip {name}: no saved CSV")
            continue
        df = pd.read_csv(csv)
        df_to_latex(df, caption, label, cfg, name)
    print(f"[make_all_tables] wrote LaTeX tables to {cfg.tables_dir()}")


if __name__ == "__main__":
    main()
