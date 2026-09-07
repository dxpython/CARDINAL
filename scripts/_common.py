"""Shared helpers for run-scripts: config, schema, fold, saving, plotting, LaTeX export."""

from __future__ import annotations

import sys
from pathlib import Path

# allow `python scripts/foo.py` to import the `src` package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config, load_config, project_root
from src.data.schema import Schema
from src.data.dataset import prepare_fold, Fold
from src.utils import set_seed
from src.experiments.common import SUMMARY_METRICS, run_model

ROOT = project_root()


def get_config(path: str | Path | None = None) -> Config:
    return load_config(path)


def get_schema(cfg: Config) -> Schema:
    return Schema.load(cfg.get("paths", {}).get("data_schema"),
                       cfg.get("paths", {}).get("column_mapping"))


def get_fold(cfg: Config, exclude_label_defining: bool = False) -> Fold:
    return prepare_fold(cfg.as_dict(), get_schema(cfg),
                        exclude_label_defining=exclude_label_defining)


def collect_metrics(res) -> dict:
    if res is None or res.error:
        return {m: float("nan") for m in SUMMARY_METRICS}
    return {m: res.metrics.get(m, float("nan")) for m in SUMMARY_METRICS}


def metrics_frame(results) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({"model": name, **collect_metrics(res)})
    return pd.DataFrame(rows)


def setup_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    if title:
        ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def save_fig(fig, name: str, cfg: Config) -> Path:
    p = cfg.figures_dir() / f"{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def df_to_latex(df: pd.DataFrame, caption: str, label: str, cfg: Config, name: str) -> Path:
    """Render a DataFrame as a booktabs LaTeX table saved under tables/."""
    p = cfg.tables_dir() / f"{name}.tex"
    formatted = df.round(3).to_string(index=False)
    body = "\n".join(f"{r}" for r in df.apply(
        lambda row: " & ".join(f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
                              for v in row), axis=1))
    header = " & ".join(str(c) for c in df.columns)
    tex = (f"% table {name} — auto-generated\n"
           f"\\begin{{table*}}[t]\n\\centering\\small\n"
           f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\begin{{tabular}}{{@{{}}l{'r' * (len(df.columns) - 1)} @{{}}}}\n\\toprule\n"
           f"{header}\\\\\\midrule\n{body}\\\\\\botrule\n"
           f"\\end{{tabular}}\n\\end{{table*}}\n")
    p.write_text(tex, encoding="utf-8")
    return p


def run_models(cfg: Config, models, exclude_label_defining: bool = False, seed: int = 20260730):
    fold = get_fold(cfg, exclude_label_defining)
    results = {}
    for m in models:
        res = run_model(m, fold, cfg.as_dict(), seed)
        results[m] = res
    return fold, results
