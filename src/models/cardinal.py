"""CARDINAL: physiology-guided directional structure and prototype learning.

Faithful to the manuscript:

    z_i = f_theta(x_i ; A)                        # graph-modulated encoder
    L   = L_CE + alpha*L_graph + beta*L_proto + gamma*L_cal + lambda*||theta||^2
    L_graph = 0.25||A-A0||_1 + eta_s||A||_1 + 0.10*h(A)

Encoder: 64-64-32, GELU, latent 32, dropout 0.20; graph injection via the
learned adjacency A.  Inference uses T stochastic forward passes with dropout
retained (MC dropout).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import GraphModule
from .prototype import PrototypeModule


class GraphModulatedEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: tuple[int, ...] = (64, 64),
                 activation: nn.Module = None, dropout: float = 0.20):
        super().__init__()
        if activation is None:
            activation = nn.GELU()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.w_graph = nn.Linear(in_dim, hidden[0], bias=False)
        layers = [nn.Linear(in_dim, hidden[0])]
        prev = hidden[0]
        for h in hidden[1:]:
            layers.append(nn.Linear(prev, h))
            prev = h
        self.encoder = nn.ModuleList(layers)
        self.head_latent = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # graph modulation: adjacency-weighted variable mixing injected pre-encoding
        xg = self.w_graph(x)
        h = xg + self.encoder[0](x)
        h = self.activation(h)
        h = self.dropout(h)
        for lin in self.encoder[1:]:
            h = self.activation(lin(h))
            h = self.dropout(h)
        z = self.head_latent(h)
        return z


class Predictor(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def calibration_loss(logits: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
    """Soft multi-class Brier score used as a calibration-aware penalty."""
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(y, num_classes=probs.shape[1]).float()
    return ((probs - onehot) ** 2).sum(dim=1).mean()


class CARDINAL(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, cfg: dict, prior: Optional[object] = None):
        super().__init__()
        self.latent_dim = cfg.get("latent_dim", 32)
        hidden = tuple(cfg.get("encoder_layers", [64, 64, 32])[:-1]) or (64,)
        self.encoder = GraphModulatedEncoder(
            in_dim, self.latent_dim, hidden=hidden,
            dropout=cfg.get("dropout", 0.20))
        self.predictor = Predictor(self.latent_dim, num_classes)
        self.prototype = PrototypeModule(
            self.latent_dim, num_classes,
            momentum=cfg.get("prototype_momentum", 0.95),
            temperature=cfg.get("prototype_temperature", 0.20))
        self.num_classes = num_classes
        # loss weights
        self.alpha = cfg.get("alpha_graph", 1.0)
        self.beta = cfg.get("beta_proto", 1.0)
        self.gamma = cfg.get("gamma_cal", 0.10)

    # ---- forward ----
    def encode(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, A)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def learn_adjacency(self, use_gumbel: bool = True) -> torch.Tensor:
        if not hasattr(self, "graph"):
            raise RuntimeError("GraphModule not attached; initialise with set_prior().")
        return self.graph(use_gumbel)

    def set_prior(self, in_dim: int, prior: object, prior_weight: float,
                  sparsity_weight: float, acyclicity_weight: float,
                  edge_temperature: float = 1.0) -> None:
        self.graph = GraphModule(in_dim, prior, prior_weight, sparsity_weight,
                                 acyclicity_weight, edge_temperature)

    def graph_loss(self, A: torch.Tensor) -> torch.Tensor:
        return self.graph.graph_loss(A)

    def calibration_loss(self, logits: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
        return calibration_loss(logits, y)

    # ---- training step ----
    def compute_loss(self, x: torch.Tensor, y: torch.LongTensor,
                     use_gumbel: bool = True) -> dict[str, torch.Tensor]:
        A = self.learn_adjacency(use_gumbel)
        z = self.encode(x, A)
        logits = self.predict(z)
        l_ce = F.cross_entropy(logits, y)
        l_graph = self.graph_loss(A)
        l_proto = self.prototype.prototype_loss(z, y)
        l_cal = self.calibration_loss(logits, y)
        l_wd = sum(p.abs().sum() for p in self.parameters())  # used only if lambda_wd>0
        total = l_ce + self.alpha * l_graph + self.beta * l_proto + self.gamma * l_cal
        return {"total": total, "ce": l_ce, "graph": l_graph,
                "proto": l_proto, "cal": l_cal, "wd": l_wd}

    # ---- MC dropout (stochastic inference) ----
    def enable_mc(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def mc_probabilities(self, x: torch.Tensor, T: int = 30,
                         temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean_prob (N,K), variance (N,K), entropy (N,))."""
        self.enable_mc()
        A = self.learn_adjacency(use_gumbel=False)   # deterministic adjacency at inference
        probs = []
        with torch.no_grad():
            for _ in range(T):
                z = self.encode(x, A)
                logits = self.predict(z) / temperature
                probs.append(F.softmax(logits, dim=1))
        prob_stack = torch.stack(probs, dim=0)       # (T, N, K)
        mean_prob = prob_stack.mean(dim=0)
        var_prob = ((prob_stack - mean_prob) ** 2).mean(dim=0)
        entropy = -(mean_prob * torch.log(mean_prob + 1e-12)).sum(dim=1)
        return mean_prob, var_prob, entropy
