"""Configuration loading and process-wide constant handling.

The single source of truth is ``configs/config.yaml`` (values mirror the paper).
The project root is auto-detected as the directory holding ``configs/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydoc import locate  # noqa: F401  (used by tests for type resolution if needed)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    """Thin dict-with-attribute access wrapper around loaded YAML."""

    def __init__(self, data: dict[str, Any], root: Path = _PROJECT_ROOT):
        self._d = data
        self.root = root

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def section(self, key: str) -> "Config":
        sub = self._d.get(key, {})
        return Config(sub, self.root) if isinstance(sub, dict) else Config({}, self.root)

    def as_dict(self) -> dict[str, Any]:
        return self._d

    def path(self, *parts: str) -> Path:
        """Resolve a path relative to the project root."""
        return self.root.joinpath(*parts)

    # ---- output handling (directories created at runtime) ----
    def out_root(self) -> Path:
        return self.path(self._d.get("output", {}).get("root", "outputs"))

    def results_dir(self) -> Path:
        d = self.out_root() / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def figures_dir(self) -> Path:
        d = self.out_root() / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def tables_dir(self) -> Path:
        d = self.out_root() / "tables"
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config from ``path`` (default: configs/config.yaml)."""
    p = Path(path) if path else _PROJECT_ROOT / "configs" / "config.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # inject the data/column-mapping paths for convenience
    data.setdefault("paths", {})
    data["paths"].setdefault("column_mapping", str(_PROJECT_ROOT / "configs" / "column_mapping.yaml"))
    data["paths"].setdefault("prior_dir", str(_PROJECT_ROOT / "configs" / "prior_graphs"))
    data["paths"].setdefault("data_schema", str(_PROJECT_ROOT / "configs" / "data_schema.yaml"))
    return Config(data)


def project_root() -> Path:
    return _PROJECT_ROOT
