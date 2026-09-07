"""Leak-safe tabular preprocessing.

A :class:`Preprocessor` is fitted on TRAINING data only, then applied to the
validation and test partitions without refitting — matching the paper's
"fold-contained preprocessing" (Supplementary Methods 2.3).

Operations:
  * continuous variables  : median imputation (train) + z-score scaling (train);
  * binary variables      : mode imputation (train), kept as {0,1};
  * categorical variables : mode imputation + one-hot encoding (train categories).

No test observation contributes to any fitted parameter.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import Schema

_NUMERIC = ("int", "float")


class Preprocessor:
    def __init__(self, schema: Schema, continuous: Iterable[str]):
        self.schema = schema
        self.columns = list(continuous)
        self._medians: dict[str, float] = {}
        self._modes: dict[str, object] = {}
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._onehot_categories: dict[str, list] = {}
        self._feature_names: list[str] = []
        self.fitted = False

    def _is_numeric(self, name: str) -> bool:
        v = self.schema.variables.get(name)
        return (v.type in _NUMERIC) if v else True

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.columns = [c for c in self.columns if c in X.columns]
        cont = [c for c in self.columns if self._is_numeric(c)]
        catg = [c for c in self.columns if not self._is_numeric(c)]
        # continuous: median + z-score
        for c in cont:
            med = X[c].median()
            mean = X[c].fillna(med).mean()
            std = X[c].fillna(med).std(ddof=0)
            self._medians[c] = med
            self._means[c] = mean
            self._stds[c] = std if std > 0 else 1.0
        # binary / categorical: mode + one-hot (fit categories on train only)
        for c in catg:
            self._modes[c] = X[c].mode().iloc[0]
            cats = sorted(pd.unique(X[c].fillna(self._modes[c]).astype(str)))
            self._onehot_categories[c] = cats
        self._feature_names = self._build_feature_names(cont, catg)
        self.fitted = True
        return self

    def _build_feature_names(self, cont: list[str], catg: list[str]) -> list[str]:
        names = list(cont)
        for c in catg:
            names += [f"{c}__{k}" for k in self._onehot_categories[c]]
        return names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self.fitted, "Preprocessor must be fitted on training data first."
        cont = [c for c in self.columns if self._is_numeric(c)]
        catg = [c for c in self.columns if not self._is_numeric(c)]
        parts: list[np.ndarray] = []
        for c in cont:
            col = X[c].astype(float).fillna(self._medians[c])
            parts.append((col - self._means[c]) / self._stds[c])
        for c in catg:
            col = X[c].fillna(self._modes[c]).astype(str)
            for k in self._onehot_categories[c]:
                parts.append((col == k).astype(float))
        if not parts:
            return np.empty((len(X), 0))
        return np.column_stack([p.to_numpy() for p in parts])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def n_features(self) -> int:
        return len(self._feature_names)
