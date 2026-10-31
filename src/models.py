"""Backbones (randomly initialized only) and the classification wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def build_backbone(name: str = "resnet18") -> nn.Module:
    """torchvision ResNet with ``weights=None``; returns pooled features of size ``out_dim``."""
    m = getattr(torchvision.models, name)(weights=None)
    m.out_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.backbone(x)))
