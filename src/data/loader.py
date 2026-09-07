"""Load patient-level data (CSV/Excel) and normalise to canonical columns.

Handles:
  * Chinese / English sheet headers via the column-mapping config;
  * sex value normalisation (男/女, male/female, M/F -> 0/1 female indicator);
  * stripping PII (patient_name) and keeping the id column for patient-level
    paired tests (DeLong / McNemar).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .schema import Schema


def _read(raw_path: str | Path) -> pd.DataFrame:
    p = Path(raw_path)
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(p)
    raise ValueError(f"Unsupported data format: {suffix}")


def normalize_sex(series: pd.Series, mapping: dict) -> pd.Series:
    """Map raw sex values to a {0,1} female indicator."""

    def _map(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v != 1)  # treat 1 as male, 0/2 as female per convention
        s = str(v).strip()
        for k, val in mapping.items():
            if s.lower() == str(k).lower():
                return val
        return np.nan

    return series.map(_map).astype(float)


def load_data(clinical_data_path: str | Path,
              schema: Schema,
              column_mapping_path: str | Path | None = None,
              sex_mapping: Optional[dict] = None) -> pd.DataFrame:
    """Load the clinical sheet and return a canonicalised DataFrame.

    Returns columns named by their canonical name. Targets (pd / dd_class /
    binary_abnormal) are retained if present; otherwise they are derived by
    :func:`src.data.labels.derive_labels`.
    """
    df = _read(clinical_data_path)
    # apply column mapping (identity for canonical headers)
    df = df.rename(columns={c: schema.canonical(c) for c in df.columns})

    # behaviour flags available in the raw file metadata
    sex_mapping = sex_mapping or _default_sex_mapping(column_mapping_path)

    if schema.canonical("性别") in df.columns:
        df["female"] = normalize_sex(df[schema.canonical("性别")], sex_mapping)

    # strip PII
    for col in ["patient_name", "姓名"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # coalesce id column
    if "patient_id" not in df.columns:
        for cand in ["序号", "住院号"]:
            if cand in df.columns:
                df["patient_id"] = df[cand]
                break
    if "patient_id" not in df.columns:
        df["patient_id"] = [f"PT-{i:04d}" for i in range(len(df))]

    return df


def _default_sex_mapping(path) -> dict:
    if not path:
        return {"男": 0, "女": 1}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cm = yaml.safe_load(fh) or {}
        return cm.get("sex_mapping", {"男": 0, "女": 1})
    except Exception:
        return {"男": 0, "女": 1}
