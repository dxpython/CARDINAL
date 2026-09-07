"""High-level leak-safe fold preparation.

``prepare_fold`` loads the clinical sheet, derives labels, obtains the fixed
split, fits preprocessing on training only, and returns torch-ready arrays for
train / validation / test.  Passing ``exclude_label_defining=True`` removes
E/e', LAVI, and TR velocity from the predictor matrix (Reviewer-5 exclusion
experiment) WITHOUT changing the label definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .loader import load_data
from .schema import Schema
from .labels import derive_labels
from .preprocessing import Preprocessor
from .split import get_split


@dataclass
class Fold:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray          # 3-class grade codes (0=normal,1=mild,2=severe)
    y_val: np.ndarray
    y_test: np.ndarray
    yb_train: np.ndarray         # binary abnormal (1) vs normal (0)
    yb_val: np.ndarray
    yb_test: np.ndarray
    pd_test: np.ndarray          # reference-index values (for patient-level stats)
    patient_ids: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    predictor_columns: list[str] = field(default_factory=list)
    label_defining_inputs: list[str] = field(default_factory=list)
    reduced: bool = False        # True if label-defining predictors were removed


def _grade_to_code(s: pd.Series) -> pd.Series:
    return s.map({"normal": 0, "mild": 1, "severe": 2}).astype(int)


def _subset_predictors(df: pd.DataFrame, preds: set[str]) -> list[str]:
    return [c for c in df.columns if c in preds]


def prepare_fold(config: dict,
                 schema: Schema,
                 data_path: Optional[str | Path] = None,
                 exclude_label_defining: bool = False) -> Fold:
    """Build the leak-controlled train/val/test fold."""

    data_path = data_path or config.get("data", {}).get("clinical_data_path",
                                                        "data/synthetic_cohort.csv")
    df = load_data(data_path, schema,
                   column_mapping_path=config.get("paths", {}).get("column_mapping"))
    df = derive_labels(df, config.get("data", {}), schema,
                       exclude_label_defining=exclude_label_defining)

    # candidate predictor columns (present in the sheet)
    preds = schema.predictor_set(exclude_label_defining=exclude_label_defining)
    predictor_columns = _subset_predictors(df, preds)
    label_defining = schema.label_defining_predictors()

    y = df["dd_class"]
    split = get_split(df, config, y)

    df = df.reset_index(drop=True)
    tr, va, te = split.train_idx, split.val_idx, split.test_idx

    pre = Preprocessor(schema, predictor_columns)
    X_train = pre.fit_transform(df.iloc[tr][predictor_columns])
    X_val = pre.transform(df.iloc[va][predictor_columns])
    X_test = pre.transform(df.iloc[te][predictor_columns])

    return Fold(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=_grade_to_code(df.iloc[tr]["dd_class"]).to_numpy(),
        y_val=_grade_to_code(df.iloc[va]["dd_class"]).to_numpy(),
        y_test=_grade_to_code(df.iloc[te]["dd_class"]).to_numpy(),
        yb_train=df.iloc[tr]["binary_abnormal"].astype(int).to_numpy(),
        yb_val=df.iloc[va]["binary_abnormal"].astype(int).to_numpy(),
        yb_test=df.iloc[te]["binary_abnormal"].astype(int).to_numpy(),
        pd_test=df.iloc[te]["pd"].astype(float).to_numpy(),
        patient_ids=df.iloc[te]["patient_id"].astype(str).to_numpy(),
        feature_names=pre.feature_names,
        predictor_columns=predictor_columns,
        label_defining_inputs=[c for c in label_defining if c in df.columns],
        reduced=exclude_label_defining,
    )
