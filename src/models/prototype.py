"""Trainable class prototypes with EMA updates and cosine-similarity alignment.

Prototype loss (paper Supplementary Eq. S4):

    L_proto = -1/N sum_i log[ exp(s(z_i, c_{y_i}) / tau)
                             / sum_k exp(s(z_i, c_k) / tau) ]

with s(.,.) cosine similarity, tau = 0.20, and each prototype updated from
TRAINING representations only via EMA with momentum 0.95.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeModule(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int,
                 momentum: float = 0.95, temperature: float = 0.20):
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        # prototypes are EMA-updated from training representations (not a
        # gradient-learned parameter), so register as buffers.
        self.register_buffer("prototypes", torch.zeros(num_classes, latent_dim))
        self.num_classes = num_classes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Cosine similarity logits (z, N_q) x (K, q) -> (N, K)."""
        zn = F.normalize(z, dim=1)
        cn = F.normalize(self.prototypes, dim=1)
        return (zn @ cn.t()) / self.temperature

    def update_ema(self, z: torch.Tensor, y: torch.LongTensor) -> None:
        """EMA update of class prototypes from (training) representations."""
        with torch.no_grad():
            for k in range(self.num_classes):
                if not (y == k).any():
                    continue
                cls_z = z[y == k]
                mean = cls_z.mean(dim=0)
                self.prototypes[k] = self.momentum * self.prototypes[k] + (
                    1.0 - self.momentum) * mean

    def prototype_loss(self, z: torch.Tensor, y: torch.LongTensor) -> torch.Tensor:
        """Supervised prototype contrastive loss (cross-entropy over prototypes)."""
        logits = self.forward(z)
        return F.cross_entropy(logits, y)

    def initialize_from(self, z: torch.Tensor, y: torch.LongTensor) -> None:
        """Set prototypes to training class means (once, after first encoding)."""
        with torch.no_grad():
            for k in range(self.num_classes):
                if not (y == k).any():
                    continue
                self.prototypes[k] = z[y == k].mean(dim=0)
