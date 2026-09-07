"""Load machine-readable prior adjacency structures for each dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

_PRIOR_DIR = Path(__file__).resolve().parents[2] / "configs" / "prior_graphs"

PRIOR_FILES = {
    "clinical": "clinical_prior.csv",
    "uci": "uci_prior.csv",
    "heart_failure": "hf_prior.csv",
    "mimic": "mimic_prior.csv",
}


def load_prior_edges(dataset: str, prior_dir: Optional[str | Path] = None) -> list[tuple[str, str]]:
    """Return list of (source, target) paired with rationale, from a prior CSV."""
    d = Path(prior_dir) if prior_dir else _PRIOR_DIR
    f = d / PRIOR_FILES[dataset]
    if not f.exists():
        raise FileNotFoundError(f"Prior adjacency not found: {f}")
    df = pd.read_csv(f)
    return list(zip(df["source"].tolist(), df["target"].tolist()))


def load_prior_with_rationale(dataset: str, prior_dir: Optional[str | Path] = None) -> pd.DataFrame:
    d = Path(prior_dir) if prior_dir else _PRIOR_DIR
    return pd.read_csv(d / PRIOR_FILES[dataset])


def outcome_edges(dataset: str, prior_dir: Optional[str | Path] = None) -> list[tuple[str, str]]:
    """Edges pointing to the outcome (used for interpretability / feature flow)."""
    return [(s, t) for s, t in load_prior_edges(dataset, prior_dir) if t == "__outcome__"]
