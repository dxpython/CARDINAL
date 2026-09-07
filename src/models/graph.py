"""Learnable directional graph with physiological prior, Gumbel--sigmoid edge
gates, prior-distance, L1 sparsity and continuous acyclicity penalties.

Graph loss (paper Supplementary Eq. S1):

    L_graph = 0.25 * ||A - A0||_1  +  eta_s * ||A||_1  +  0.10 * h(A)

with the continuous acyclicity term (Eq. S2):

    h(A) = tr[exp(A .* A)] - d
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False,
                   eps: float = 1e-20) -> torch.Tensor:
    """Categorical reparameterised sigmoid for learnable edge gates."""
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)
    y = torch.sigmoid((logits + g) / temperature)
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard - y.detach() + y
    return y


def acyclicity(A: torch.Tensor) -> torch.Tensor:
    """h(A) = tr[exp(A .* A)] - d.  Zero iff A is a valid DAG (NOTEARS)."""
    A2 = A * A
    return torch.matrix_exp(A2).diagonal().sum() - A.shape[0]


def build_prior_from_edges(edges: list[tuple[str, str]], feature_names: list[str]) -> np.ndarray:
    """Build an (d, d) prior adjacency over encapsulated feature columns.

    Prior edges referencing a predictor node are matched to encapsulated column
    names (``name`` or ``name__level``).  Edges whose target is the outcome
    (``__outcome__``) are excluded from the predictor--predictor matrix here and
    are tracked separately for interpretability.
    """
    index = {name: i for i, name in enumerate(feature_names)}
    d = len(feature_names)
    A0 = np.zeros((d, d), dtype=np.float32)

    def match(name):
        if name in index:
            return index[name]
        buckets = [i for n, i in index.items() if n.startswith(name + "__")]
        return buckets[0] if len(buckets) == 1 else None

    for src, tgt in edges:
        if tgt == "__outcome__":
            continue
        i, j = match(src), match(tgt)
        if i is not None and j is not None:
            A0[i, j] = 1.0
    return A0


class GraphModule(nn.Module):
    """Learnable directional adjacency, refined from a dataset-specific prior."""

    def __init__(self, d: int, prior: np.ndarray,
                 prior_weight: float = 0.25,
                 sparsity_weight: float = 0.05,
                 acyclicity_weight: float = 0.10,
                 edge_temperature: float = 1.0):
        super().__init__()
        self.d = d
        self.prior_weight = prior_weight
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
        self.edge_temperature = edge_temperature
        self.register_buffer("A0", torch.tensor(prior, dtype=torch.float32))
        # initialise edge logits so A starts near A0
        init = torch.where(torch.tensor(prior) > 0.5, torch.tensor(2.5), torch.tensor(-2.5))
        self.edge_logits = nn.Parameter(init)

    def forward(self, use_gumbel: bool = True) -> torch.Tensor:
        """Return the refined adjacency A (d, d), optionally Gumbel-sampled."""
        if use_gumbel and self.training:
            A = gumbel_sigmoid(self.edge_logits, self.edge_temperature)
        else:
            A = torch.sigmoid(self.edge_logits)
        return A

    def regularized_A(self, A: torch.Tensor) -> torch.Tensor:
        """Noise-free adjacency used for penalties (matches prior-distance semantics)."""
        return torch.sigmoid(self.edge_logits)

    def graph_loss(self, A: torch.Tensor) -> torch.Tensor:
        A0 = self.A0
        prior_dist = (A - A0).abs().sum()
        sparsity = self.sparsity_weight * A.abs().sum()
        dag = self.acyclicity_weight * acyclicity(A)
        return self.prior_weight * prior_dist + sparsity + dag
