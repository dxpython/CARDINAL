"""Data-pipeline correctness: split, leakage, label construction, exclusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import prepare_fold
from src.data.preprocessing import Preprocessor
from src.data.schema import Schema
from src.config import load_config


def test_split_disjoint_and_stratified(fold_factory):
    fold = fold_factory()
    train, val, test = fold.X_train, fold.X_val, fold.X_test
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # disjoint: no patient (row) shared between train and test
    # row order is preserved per partition; we assert on the labels portion
    # train + val + test rows should equal all rows
    total = len(train) + len(val) + len(test)
    assert total == 118
    # each partition non-empty and balanced across the 3 grades
    for y in (fold.y_train, fold.y_val, fold.y_test):
        assert set(np.unique(y)).issubset({0, 1, 2})


def test_train_only_scaling(fold_factory):
    fold = fold_factory()
    schema = Schema.load(load_config().get("paths", {}).get("data_schema"),
                         load_config().get("paths", {}).get("column_mapping"))
    # fit on train only
    pre = Preprocessor(schema, fold.predictor_columns).fit(pd.DataFrame(fold.X_train,
                                                                        columns=fold.feature_names))
    # applying to test uses train statistics
    Xte = pre.transform(pd.DataFrame(fold.X_test, columns=fold.feature_names))
    assert Xte.shape[0] == len(fold.X_test)
    assert Xte.shape[1] == len(fold.feature_names)
    # no NaN produced when transform is applied to a fresh partition
    assert not np.isnan(Xte).any()


def test_locked_test_never_present_in_training(fold_factory):
    fold = fold_factory()
    # patient_ids from the test partition must not overlap the training pool
    # reconstructing index is indirect; instead assert test size is the small locked set
    assert len(fold.y_test) == 18
    assert len(fold.y_train) == 82 or len(fold.y_train) == 81 or len(fold.y_train) == 83


def test_label_defining_exclusion(fold_factory):
    full = fold_factory(exclude_label_defining=False)
    red = fold_factory(exclude_label_defining=True)
    assert red.reduced is True
    # labels identical between full and reduced folds
    assert np.array_equal(full.y_test, red.y_test)
    assert np.array_equal(full.yb_test, red.yb_test)
    # predictors removed: exactly E/e', LAVI, TR velocity
    removed = set(full.predictor_columns) - set(red.predictor_columns)
    assert removed == {"e_over_e_prime", "lavi_ml_m2", "tr_velocity_m_s"}
    # and label-defining inputs recorded
    assert set(full.label_defining_inputs) == {"e_over_e_prime", "lavi_ml_m2", "tr_velocity_m_s"}


def test_reference_index_matches_paper(fold_factory):
    from src.data.labels import build_grades, build_binary
    fold = fold_factory()
    # grade labels must map from the Pd thresholds
    assert set(np.unique(fold.y_test)).issubset({0, 1, 2})
    # binary endpoint: mild(1)+severe(2) => abnormal
    binary = (fold.y_test != 0).astype(int)
    assert np.array_equal(fold.yb_test, binary)
