"""Small-sample evaluation on repeated stratified subsamples of the training pool.

Validation (18) and locked test (18) are NEVER resampled; only the fixed
82-case training pool is subsampled.  Because three-class prototype learning
requires >=3 cases per class, the nominal 5% and 10% settings both use 9 cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import get_config, get_fold, save_fig, setup_ax
from src.experiments.common import train_cardinal, evaluate_cardinal, SUMMARY_METRICS
from src.data.preprocessing import Preprocessor
from src.data.labels import derive_labels
from src.utils import set_seed, save_csv


def main():
    cfg = get_config()
    seed = int(cfg.get("seed"))
    fold = get_fold(cfg)
    fractions = cfg.get("small_sample", {}).get("fractions", [0.05, 0.10, 0.20, 0.40, 1.0])
    min_per_class = cfg.get("small_sample", {}).get("min_per_class", 3)
    n_repeats = cfg.get("small_sample", {}).get("repeats", 20)
    model_cfg = dict(cfg.get("model", {}))

    # recover the fixed training pool rows (already fitted on the full train pool)
    rows = []
    for frac in fractions:
        for rep in range(n_repeats):
            set_seed(seed + rep)
            # stratified subsample of the fixed train pool
            Xtr, ytr, Xva, yva = subsample_train(fold, frac, min_per_class, seed + rep)
            if Xtr is None:
                continue
            m, _, _ = train_cardinal_from_arrays(Xtr, ytr, Xva, yva, fold, cfg, model_cfg,
                                                 seed + rep)
            if m is None:
                continue
            res = evaluate_cardinal(m, fold, cfg.as_dict(), seed + rep)
            rows.append({"training_fraction": float(frac),
                         "actual_training_n": len(ytr),
                         "repeat": rep, "test_auc": res.metrics.get("auc", float("nan"))})

    df = pd.DataFrame(rows)
    if df.empty:
        print("[run_small_sample] no runs completed (check data / dependencies)")
        return
    # mean and 95% CI per fraction
    summ = (df.groupby("training_fraction")["test_auc"]
              .agg(["mean", "std", "count"]).reset_index())
    summ["ci_low"] = summ["mean"] - 1.96 * summ["std"] / np.sqrt(summ["count"])
    summ["ci_high"] = summ["mean"] + 1.96 * summ["std"] / np.sqrt(summ["count"])
    save_csv(summ, cfg.results_dir() / "small_sample_summary.csv")
    save_csv(df, cfg.results_dir() / "small_sample_results.csv")

    fig, ax = setup_ax(None, "Small-sample performance", "Training cases", "ROC-AUC")
    ax.errorbar(summ["actual_training_n"], summ["mean"], yerr=1.96 * summ["std"] / np.sqrt(summ["count"]),
                fmt="-o", color="tab:blue", capsize=4)
    save_fig(fig, "06_small_sample_performance", cfg)
    print("[run_small_sample] wrote small-sample results + figure")


def subsample_train(fold, fraction, min_per_class, seed):
    rng = np.random.default_rng(int(seed))
    y = fold.y_train
    # per-class sample budget
    n_target = max(min_per_class, int(round(len(y) * fraction)))
    idxs = []
    for k in range(3):
        pool = np.where(y == k)[0]
        n_k = min(len(pool), n_target if fraction >= 1.0 else max(min_per_class, int(round(len(pool) * fraction))))
        # for nominal fractions, keep min_per_class feasibility
        if fraction < 1.0 and len(pool) <= min_per_class:
            n_k = len(pool)
        idxs.append(rng.choice(pool, size=n_k, replace=False))
    chosen = np.concatenate(idxs)
    return fold.X_train[chosen], y[chosen], fold.X_val, fold.y_val


def train_cardinal_from_arrays(Xtr, ytr, Xva, yva, fold, cfg, model_cfg, seed):
    from src.experiments.common import build_prior
    import torch
    from src.models.cardinal import CARDINAL
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from src.utils import get_device

    device = get_device()
    set_seed(seed)
    try:
        model = CARDINAL(Xtr.shape[1], 3, model_cfg).to(device)
        model.set_prior(Xtr.shape[1], build_prior(fold), model_cfg.get("prior_weight", 0.25),
                        model_cfg.get("sparsity_weight", 0.05),
                        model_cfg.get("acyclicity_weight", 0.10))
        Xte_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
        yte_t = torch.tensor(ytr, dtype=torch.long).to(device)
        opt = torch.optim.AdamW(model.parameters(),
                                lr=model_cfg.get("learning_rate", 1e-3),
                                weight_decay=model_cfg.get("weight_decay", 1e-4))
        dl = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=model_cfg.get("batch_size", 16),
                        shuffle=True)
        best, patience = 1e9, 0
        for epoch in range(model_cfg.get("max_epochs", 200)):
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                losses = model.compute_loss(xb, yb)
                losses["total"].backward(); opt.step()
            model.eval()
            with torch.no_grad():
                A = model.learn_adjacency(use_gumbel=False)
                ztr = model.encode(Xte_t, A)
            model.prototype.update_ema(ztr, yte_t)
            with torch.no_grad():
                vl = nn.functional.cross_entropy(model.predict(model.encode(torch.tensor(Xva, dtype=torch.float32).to(device), A)), torch.tensor(yva, dtype=torch.long).to(device))
            patience = patience + 1 if vl.item() < best else 0
            best = min(best, vl.item())
            if patience >= model_cfg.get("early_stopping_patience", 20):
                break
        return model, None, None
    except Exception as e:  # noqa: BLE001
        print("  small-sample run failed:", e)
        return None, None, None


if __name__ == "__main__":
    main()
