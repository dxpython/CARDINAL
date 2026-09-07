"""Public tabular robustness benchmarks: UCI Heart Disease, Heart Failure
Clinical Records, MIMIC-IV cardiac subset.

These are general-purpose tabular benchmarks with their own variable systems,
endpoints, and dataset-specific prior graphs — NOT external clinical validation
of diastolic dysfunction.  Raw benchmark data are NOT bundled; if a file is
missing from ``data/`` a dependency gap is recorded instead of fabricating
results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from _common import get_config, save_fig, setup_ax
from src.graph.prior import load_prior_edges
from src.models.cardinal import CARDINAL
from src.models.graph import build_prior_from_edges
from src.models.baselines import get_baseline
from src.metrics.discrimination import discrimination_metrics
from src.metrics.calibration import ece, brier_score, nll_binary
from src.utils import set_seed, save_csv

# dataset -> (key, predictor columns, target column, prior file)
BENCHMARKS = {
    "UCI Heart Disease": ("uci", ["age", "sex", "cp", "trestbps", "chol", "fbs",
                                  "restecg", "thalach", "exang", "oldpeak", "slope"],
                          "target"),
    "Heart Failure Clinical Records": ("hf", ["age", "anaemia", "creatinine_phosphokinase",
                                              "diabetes", "ejection_fraction",
                                              "high_blood_pressure", "platelets",
                                              "serum_creatinine", "serum_sodium", "sex",
                                              "smoking", "time"], "DEATH_EVENT"),
    "MIMIC-IV": ("mimic", ["age", "heart_rate", "mean_arterial_pressure", "creatinine",
                           "lactate", "severity_score", "organ_dysfunction",
                           "comorbidity_burden"], "mortality"),
}


def load_benchmark(key: str, cfg) -> tuple[pd.DataFrame, pd.Series] | None:
    candidates = [cfg.path("data", f"{key}_benchmark.csv"),
                  cfg.path("data", f"{key}.csv")]
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c)
            return df, df.iloc[:, -1]
    # document the gap
    gap = cfg.results_dir() / "benchmark_data_gaps.json"
    gaps = json.loads(gap.read_text()) if gap.exists() else {}
    gaps[key] = "raw benchmark data not present in code/data/"
    gap.write_text(json.dumps(gaps, indent=2))
    return None


def train_cardinal_binary(X, y, Xva, yva, prior, cfg):
    model_cfg = dict(cfg.get("model", {}))
    model = CARDINAL(X.shape[1], 2, model_cfg)
    model.set_prior(X.shape[1], prior, model_cfg.get("prior_weight", 0.25),
                    model_cfg.get("sparsity_weight", 0.05),
                    model_cfg.get("acyclicity_weight", 0.10))
    device = "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=16, shuffle=True)
    best, patience = 1e9, 0
    for epoch in range(200):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            losses = model.compute_loss(xb, yb)
            losses["total"].backward(); opt.step()
        model.eval()
        with torch.no_grad():
            A = model.learn_adjacency(use_gumbel=False)
            z = model.encode(torch.tensor(Xva, dtype=torch.float32), A)
            vl = torch.nn.functional.cross_entropy(model.predict(z), torch.tensor(yva, dtype=torch.long))
        patience = patience + 1 if vl.item() < best else 0
        best = min(best, vl.item())
        if patience >= 20:
            break
    model.eval()
    return model


def main():
    cfg = get_config()
    seed = int(cfg.get("seed"))
    out_rows = []
    for name, (key, predictors, target) in BENCHMARKS.items():
        loaded = load_benchmark(key, cfg)
        if loaded is None:
            out_rows.append({"dataset": name, "model": "NA", "auc": float("nan"),
                             "auprc": float("nan"), "brier": float("nan"),
                             "note": "data gap"})
            continue
        df, y = loaded
        avail_preds = [p for p in predictors if p in df.columns]
        X = df[avail_preds].select_dtypes("number").fillna(df[avail_preds].median()).to_numpy(float)
        y = y.astype(int).to_numpy()
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
        # small validation from train for CARDINAL early stopping
        Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed, stratify=ytr)

        prior = build_prior_from_edges(load_prior_edges(key), avail_preds)
        mod = train_cardinal_binary(Xtr2, ytr2, Xva, yva, prior, cfg)
        with torch.no_grad():
            mean_prob, _, _ = mod.mc_probabilities(torch.tensor(Xte, dtype=torch.float32), T=30)
        p_card = mean_prob.numpy()[:, 1]
        out_rows.append({"dataset": name, "model": "CARDINAL",
                         **{k: v for k, v in discrimination_metrics(yte, p_card).items()
                            if k in ("auc", "auprc", "accuracy")},
                         "brier": brier_score(yte, p_card)})
        for bname in ["TabPFN-tuned"]:
            try:
                bl = get_baseline(bname)
                bl.fit(Xtr2, ytr2)
                p = bl.predict_proba_abnormal(Xte)
                out_rows.append({"dataset": name, "model": bname,
                                 **{k: v for k, v in discrimination_metrics(yte, p).items()
                                    if k in ("auc", "auprc", "accuracy")},
                                 "brier": brier_score(yte, p)})
            except Exception as e:  # noqa: BLE001
                out_rows.append({"dataset": name, "model": bname, "auc": float("nan"),
                                 "auprc": float("nan"), "brier": float("nan"),
                                 "note": f"{type(e).__name__}: {e}"})

    df_out = pd.DataFrame(out_rows)
    save_csv(df_out, cfg.results_dir() / "benchmark_summary.csv")
    print("[run_public_benchmarks] wrote benchmark summary; missing-data datasets "
          "recorded in benchmark_data_gaps.json")


if __name__ == "__main__":
    main()
