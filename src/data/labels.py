"""Reference index (Pd), grade, and binary-endpoint construction.

The reference index is defined exactly as in the paper:

    Pd = 0.45 * z(E/e') + 0.35 * z(LAVI) + 0.20 * z(TR velocity)

with z-scores from the cohort reference distribution, and three ordered grades:

    normal      if Pd <  -0.35
    mild        if -0.35 <= Pd < 0.70
    severe      if  Pd >= 0.70

The binary endpoint combines mild+severe as abnormal vs normal.  The three
label-defining inputs (E/e', LAVI, TR velocity) are used ONLY to build these
targets; they are separately tracked so the Reviewer-5 exclusion experiment can
remove them from the predictor matrix without touching the label definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import Schema


def _zscore_from_col(values: pd.Series) -> pd.Series:
    """Reference-distribution z-score (mean/std of the passed series)."""
    mu = values.mean()
    sd = values.std(ddof=0)
    return (values - mu) / sd


def build_pd(df: pd.DataFrame, ref_index: dict[str, float]) -> pd.Series:
    """Compute Pd from E/e', LAVI, TR velocity if not already present.

    z-scores use the full cohort reference distribution (paper Methods 2.1).
    """
    if "pd" in df.columns and df["pd"].notna().all():
        return df["pd"].astype(float)
    terms = []
    for col, w in ref_index.items():
        if col not in df.columns:
            raise KeyError(f"Missing reference-index variable: {col}")
        terms.append(w * _zscore_from_col(df[col].astype(float)))
    return sum(terms)


def build_grades(pd_series: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    mild_t = thresholds.get("mild", -0.35)
    severe_t = thresholds.get("severe", 0.70)
    out = pd.Series(
        np.select(
            [pd_series < mild_t, pd_series < severe_t],
            ["normal", "mild"],
            default="severe",
        ),
        index=pd_series.index,
        name="dd_class",
    )
    return out


def build_binary(dd_class: pd.Series, abnormal_threshold: float = 0.50) -> pd.Series:
    """binary endpoint: mild+severe combined as abnormal (1) vs normal (0)."""
    # in the paper the binary endpoint simply marks abnormal == mild or severe
    return (dd_class != "normal").astype(int)


def derive_labels(df: pd.DataFrame,
                  config: dict,
                  schema: Schema,
                  exclude_label_defining: bool = False) -> pd.DataFrame:
    """Derive pd / dd_class / binary_abnormal and record used label-defining inputs.

    ``exclude_label_defining`` does NOT change the labels. It only controls which
    predictors are removed from the model input matrix (handled by the pipeline);
    labels are identical in both settings by design.
    """
    ref_index = config.get("ref_index", {})
    have_pd = config.get("has_pd_column", False)
    if have_pd and "pd" in df.columns:
        pd_series = df["pd"].astype(float)
    else:
        pd_series = build_pd(df, ref_index)

    # If a pre-computed grade column is present and valid, keep it.
    grade_col = config.get("target_grades_column", None)
    if grade_col and grade_col in df.columns:
        dd_class = df[grade_col].astype(str).str.lower()
    else:
        dd_class = build_grades(pd_series, config.get("grade_thresholds", {}))

    out = df.copy()
    out["pd"] = pd_series.astype(float)
    out["dd_class"] = dd_class
    out["binary_abnormal"] = build_binary(dd_class, config.get("abnormal_threshold", 0.50))
    # track which label-defining inputs were used
    out.attrs["label_defining_inputs"] = [c for c in schema.label_defining_predictors() if c in out.columns]
    return out
