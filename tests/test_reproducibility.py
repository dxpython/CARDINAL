"""Reproducibility smoke test: seeded CARDINAL training is deterministic."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.utils import set_seed
from src.config import load_config
from src.experiments.common import build_prior
from src.models.cardinal import CARDINAL


def test_set_seed_deterministic(fold_factory):
    a = train_small(fold_factory(), load_config(), seed=11)
    b = train_small(fold_factory(), load_config(), seed=11)
    assert np.array_equal(a, b)


def test_set_seed_changes_with_seed(fold_factory):
    a = train_small(fold_factory(), load_config(), seed=11)
    b = train_small(fold_factory(), load_config(), seed=99)
    assert not np.array_equal(a, b)


def train_small(fold, cfg, seed):
    set_seed(seed)
    model_cfg = {"latent_dim": 32, "encoder_layers": [64, 64, 32], "dropout": 0.20,
                 "prototype_momentum": 0.95, "prototype_temperature": 0.20}
    model = CARDINAL(fold.X_train.shape[1], 3, model_cfg)
    model.set_prior(fold.X_train.shape[1], build_prior(fold), 0.25, 0.05, 0.10)
    X = torch.tensor(fold.X_train[:32], dtype=torch.float32)
    y = torch.tensor(fold.y_train[:32], dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(3):
        opt.zero_grad()
        losses = model.compute_loss(X, y)
        losses["total"].backward()
        opt.step()
    with torch.no_grad():
        A = model.learn_adjacency(use_gumbel=False)
        z = model.encode(X, A)
    return z.detach().numpy()
