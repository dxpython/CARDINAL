"""Directional-graph analysis relative to the prespecified reference structure.

Exports the learnable prior, the learned adjacency, edge-selection frequency and
direction agreement across repeated runs, compares CARDINAL against PC, NOTEARS
and DAG-GNN, and reports precision / recall / F1 / SHD as reference-concordance
(not causal truth).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _common import get_config, get_fold, save_fig, setup_ax
from src.graph.prior import load_prior_edges
from src.graph.compare import concordance
from src.graph.stability import edge_frequencies, direction_agreement
from src.graph.structure import run_structure_learning
from src.models.cardinal import CARDINAL
from src.models.graph import build_prior_from_edges
from src.experiments.common import build_prior, train_cardinal
from src.utils import set_seed, save_csv


def learned_A(model: CARDINAL) -> np.ndarray:
    with torch.no_grad():
        model.eval()
        A = torch.sigmoid(model.graph.edge_logits).numpy()
    return (A > 0.5).astype(int)


def main():
    cfg = get_config()
    seed = int(cfg.get("seed"))
    fold = get_fold(cfg)
    model_cfg = dict(cfg.get("model", {}))
    prior = build_prior(fold)                       # reference A0 over features
    feature_names = fold.feature_names

    rows = []
    # --- CARDINAL learned-graph concordance (mean over repeated seeds) ---
    adjs = []
    for r in range(cfg.get("num_repeated_runs", 20)):
        set_seed(seed + r)
        m, _, _ = train_cardinal(fold, cfg.as_dict(), model_cfg, seed=seed + r)
        adjs.append(learned_A(m))
    card_conc = concordance(np.mean(adjs, axis=0) > 0.5, prior)
    rows.append({"method": "CARDINAL", **card_conc})
    freq = edge_frequencies(adjs)
    direction_agree = direction_agreement(adjs, prior)

    # --- PC (self-contained) & NOTEARS / DAG-GNN (guarded) ---
    import pandas as pd
    X = pd.DataFrame(fold.X_train, columns=fold.feature_names)
    for method in cfg.get("graph_analysis", {}).get("structure_methods", ["PC", "NOTEARS", "DAG-GNN"]):
        if method == "CARDINAL":
            continue
        try:
            est = run_structure_learning(X, method)
        except (ImportError, Exception) as e:  # noqa: BLE001
            rows.append({"method": method, "precision": float("nan"), "recall": float("nan"),
                         "f1": float("nan"), "shd": float("nan"),
                         "note": f"{type(e).__name__}: {e}"})
            continue
        rows.append({"method": method, **concordance(est, prior)})

    df = pd.DataFrame(rows)
    save_csv(df, cfg.results_dir() / "graph_stability.csv")

    # exported prior + edge frequencies
    pd.DataFrame({"feature": feature_names}).to_csv(
        cfg.results_dir() / "feature_names.csv", index=False)
    freq_df = pd.DataFrame(freq, index=feature_names, columns=feature_names)
    freq_df.to_csv(cfg.results_dir() / "edge_selection_frequency.csv")
    save_csv(pd.DataFrame({"direction_agreement": [direction_agree]}),
             cfg.results_dir() / "direction_agreement.csv")

    # figure: comparative ROC of structure methods
    methods = df["method"].tolist()
    f1 = df["f1"].tolist()
    fig, ax = setup_ax(None, "Graph recovery (reference concordance)", "Method", "F1")
    ax.bar(methods, [np.nan if isinstance(v, str) else v for v in f1])
    save_fig(fig, "10_graph_stability", cfg)
    print("[run_graph_analysis] wrote graph analysis results + figure")


if __name__ == "__main__":
    main()
