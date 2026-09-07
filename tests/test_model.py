"""CARDINAL component tests: prior shape, prototype EMA, MC stochasticity, loss."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.cardinal import CARDINAL
from src.models.prototype import PrototypeModule
from src.models.graph import acyclicity, build_prior_from_edges
from src.data.schema import Schema
from src.config import load_config


def _model_cfg():
    return {"latent_dim": 32, "encoder_layers": [64, 64, 32], "dropout": 0.20,
            "prototype_momentum": 0.95, "prototype_temperature": 0.20,
            "alpha_graph": 1.0, "beta_proto": 1.0, "gamma_cal": 0.10}


def test_prior_shape_and_values():
    cfg = load_config()
    schema = Schema.load(cfg.get("paths", {}).get("data_schema"))
    edges = [("age_years", "hypertension"), ("hypertension", "lavi_ml_m2")]
    names = ["age_years", "hypertension", "lavi_ml_m2", "bmi_kg_m2"]
    A0 = build_prior_from_edges(edges, names)
    assert A0.shape == (4, 4)
    assert set(np.unique(A0)) <= {0.0, 1.0}
    assert np.allclose(np.diag(A0), 0.0)      # no self-loops
    assert A0[0, 1] == 1.0 and A0[1, 2] == 1.0  # age->hypertension, hypertension->lavi


def test_acyclicity_zero_for_identity():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    h = acyclicity(torch.tensor(A))
    assert abs(h.item()) < 1e-4


def test_prototype_ema_update():
    proto = PrototypeModule(latent_dim=4, num_classes=2, momentum=0.95, temperature=0.20)
    proto.prototypes[0] = torch.zeros(4)
    z = torch.tensor([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]])
    y = torch.tensor([0, 0])
    proto.initialize_from(z, y)
    assert torch.allclose(proto.prototypes[0], torch.tensor([2.0] * 4))
    proto.update_ema(z, y)  # momentum 0.95 -> stays ~2.0
    assert abs(float((proto.prototypes[0] - torch.tensor([2.0] * 4)).abs().max())) < 1e-3


def test_mc_dropout_stochastic(fold_factory):
    fold = fold_factory()
    cfg = load_config()
    model = CARDINAL(fold.X_train.shape[1], 3, _model_cfg())
    from src.experiments.common import build_prior
    model.set_prior(fold.X_train.shape[1], build_prior(fold), 0.25, 0.05, 0.10)
    X = torch.tensor(fold.X_test[:5], dtype=torch.float32)
    mean_p, var_p, entropy = model.mc_probabilities(X, T=30)
    assert mean_p.shape == (5, 3)
    assert var_p.shape == (5, 3)
    assert entropy.shape == (5,)
    # MC dropout with dropout active must produce non-degenerate variance
    assert var_p.max().item() > 0.0


def test_cardinal_loss_runs(fold_factory):
    fold = fold_factory()
    cfg = load_config()
    model = CARDINAL(fold.X_train.shape[1], 3, _model_cfg())
    from src.experiments.common import build_prior
    model.set_prior(fold.X_train.shape[1], build_prior(fold), 0.25, 0.05, 0.10)
    X = torch.tensor(fold.X_train[:16], dtype=torch.float32)
    y = torch.tensor(fold.y_train[:16], dtype=torch.long)
    losses = model.compute_loss(X, y)
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["graph"])
    assert torch.isfinite(losses["proto"])
