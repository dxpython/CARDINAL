"""Variable schema: what is a predictor, a label-defining input, a target, an id.

Loads ``configs/data_schema.yaml`` and exposes helpers used across the pipeline,
especially the Reviewer-5 "label-defining feature exclusion" logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Variable:
    name: str
    type: str = "continuous"
    role: str = "predictor"
    label_defining: bool = False
    synthetic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Schema:
    def __init__(self, variables: dict[str, Variable], mapping: dict[str, str]):
        self.variables = variables
        self.mapping = mapping  # data-sheet header -> canonical name

    @classmethod
    def load(cls, path: str | Path, column_mapping_path: str | Path | None = None) -> "Schema":
        p = Path(path)
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        vars_def = raw.get("variables", {})
        variables: dict[str, Variable] = {}
        for name, meta in vars_def.items():
            meta = meta or {}
            variables[name] = Variable(
                name=name,
                type=meta.get("type", "continuous"),
                role=meta.get("role", "predictor"),
                label_defining=bool(meta.get("label_defining", False)),
                synthetic=bool(meta.get("synthetic", False)),
                extra=meta,
            )
        mapping: dict[str, str] = {}
        if column_mapping_path:
            with open(column_mapping_path, "r", encoding="utf-8") as fh2:
                cm = yaml.safe_load(fh2) or {}
            mapping = cm.get("mapping", {})
        return cls(variables, mapping)

    # ---- categorization ----
    def predictors(self) -> list[str]:
        return [v.name for v in self.variables.values()
                if v.role == "predictor" and not v.synthetic]

    def label_defining_predictors(self) -> list[str]:
        return [v.name for v in self.variables.values()
                if v.role == "predictor" and v.label_defining and not v.synthetic]

    def targets(self) -> list[str]:
        return [v.name for v in self.variables.values() if v.role == "target"]

    def predictor_set(self, exclude_label_defining: bool = False) -> set[str]:
        """Predictor columns to feed the model.

        If ``exclude_label_defining`` is True, E/e', LAVI, and TR velocity are
        removed from the predictor matrix (Reviewer-5 exclusion experiment). The
        label targets (Pd / grade / binary) are never inputs in either case.
        """
        preds = set(self.predictors())
        if exclude_label_defining:
            preds -= set(self.label_defining_predictors())
        return preds

    def canonical(self, header: str) -> str:
        """Map a data-sheet header to its canonical name (identity if unmapped)."""
        return self.mapping.get(header, header)

    def deserialize(self) -> dict[str, Any]:
        return {
            "variables": {k: {"type": v.type, "role": v.role,
                              "label_defining": v.label_defining,
                              }
                          for k, v in self.variables.items()},
            "mapping": self.mapping,
        }
