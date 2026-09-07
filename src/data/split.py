"""Fixed, stratified train/validation/test allocation.

The paper uses a 70/15/15 stratified split producing an 82-case training pool,
an 18-case validation set, and an 18-case locked test set for the 118-case
cohort (Methods 2.6).  If the data already carries a ``split`` column (e.g. the
synthetic smoke-test cohort), that partition is reused verbatim so results stay
consistent with the fixed split.  A fresh split is computed when absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def size(self) -> dict[str, int]:
        return {"train": len(self.train_idx), "validation": len(self.val_idx),
                "test": len(self.test_idx)}


def make_stratified_split(y: pd.Series,
                          train_frac: float = 0.70,
                          val_frac: float = 0.15,
                          seed: int = 20260730,
                          ) -> Split:
    """Stratified split preserving the class distribution of ``y`` (grades).

    Two successive stratified splits give a 70/15/15 allocation while keeping
    every partition class-balanced with respect to the three grades.
    """
    idx = np.arange(len(y))
    rest_frac = 1.0 - train_frac
    # stage 1: separate the training pool (70%) from the rest (30%)
    train_idx, rest_idx = train_test_split(
        idx, test_size=rest_frac, random_state=seed, stratify=y)
    # stage 2: split the rest 50/50 into validation and test (=> 15% / 15%)
    rest_y = y.iloc[rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, random_state=seed, stratify=rest_y)
    return Split(train_idx, val_idx, test_idx)


def from_split_column(df: pd.DataFrame, maps: Optional[dict] = None) -> Split:
    maps = maps or {"train": "train", "validation": "val", "val": "val", "test": "test"}  # noqa: E501
    train = np.where(df["split"].astype(str).str.lower().isin(["train"]))[0]
    val = np.where(df["split"].astype(str).str.lower().isin(["validation", "val"]))[0]
    test = np.where(df["split"].astype(str).str.lower().isin(["test"]))[0]
    return Split(train, val, test)


def get_split(df: pd.DataFrame, config: dict, y: pd.Series) -> Split:
    """Return the partition, preferring a stored ``split`` column if present."""
    if "split" in df.columns:
        return from_split_column(df)
    sc = config.get("split", {})
    return make_stratified_split(
        y, sc.get("train_frac", 0.70), sc.get("val_frac", 0.15),
        seed=int(config.get("seed", 20260730)))
