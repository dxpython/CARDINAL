import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.config import load_config
from src.data.schema import Schema
from src.data.dataset import prepare_fold


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def schema(cfg):
    return Schema.load(cfg.get("paths", {}).get("data_schema"),
                       cfg.get("paths", {}).get("column_mapping"))


@pytest.fixture(scope="session")
def fold_factory():
    def _make(exclude_label_defining=False):
        cfg = load_config()
        sch = Schema.load(cfg.get("paths", {}).get("data_schema"),
                          cfg.get("paths", {}).get("column_mapping"))
        return prepare_fold(cfg.as_dict(), sch, exclude_label_defining=exclude_label_defining)
    return _make
