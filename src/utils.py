"""Shared utilities: seed control, device info, path/save helpers, tabular data IO."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    """Deterministic seeding for python, numpy, and torch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  
        pass


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"device": get_device()}
    try:
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import platform

        info["platform"] = platform.platform()
        info["python"] = platform.python_version()
    except Exception:
        pass
    return info


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
    return p


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return p


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(p, **arrays)
    return p


def chunked(seq: Iterable, size: int) -> Iterable[list]:
    """Yield successive chunks of ``seq`` of length ``size``."""
    out = []
    for item in seq:
        out.append(item)
        if len(out) == size:
            yield out
            out = []
    if out:
        yield out
