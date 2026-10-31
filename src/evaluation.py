"""Feature extraction, kNN / linear probes on frozen features, and (TTA) prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import eval_view


@torch.no_grad()
def extract(backbone: nn.Module, x8: torch.Tensor, size: int, bs: int = 256) -> torch.Tensor:
    was_training = backbone.training
    backbone.eval()
    feats = []
    for i in range(0, len(x8), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(backbone(eval_view(x8[i:i + bs], size).contiguous(memory_format=torch.channels_last)).float())
    backbone.train(was_training)
    return torch.cat(feats)


@torch.no_grad()
def knn_acc(train_f, train_y, val_f, val_y, k: int = 20, temperature: float = 0.07) -> float:
    """Weighted cosine kNN (as in DINO / InstDisc)."""
    tf, vf = F.normalize(train_f, dim=1), F.normalize(val_f, dim=1)
    sim, idx = (vf @ tf.t()).topk(k, dim=1)
    votes = torch.zeros(len(vf), int(train_y.max()) + 1, device=vf.device)
    votes.scatter_add_(1, train_y[idx], (sim / temperature).exp())
    return (votes.argmax(1) == val_y).float().mean().item()


def linear_probe(train_f, train_y, val_f, val_y, steps: int = 500, wd: float = 1e-3) -> float:
    """Full-batch multinomial logistic regression on standardized frozen features."""
    mu, sd = train_f.mean(0), train_f.std(0) + 1e-6
    tf, vf = (train_f - mu) / sd, (val_f - mu) / sd
    lin = nn.Linear(tf.shape[1], int(train_y.max()) + 1).to(tf.device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(lin(tf), train_y).backward()
        opt.step()
    with torch.no_grad():
        return (lin(vf).argmax(1) == val_y).float().mean().item()


def probes(backbone: nn.Module, data: dict, size: int) -> dict:
    tf = extract(backbone, data["train_x"], size)
    vf = extract(backbone, data["val_x"], size)
    return {"knn": knn_acc(tf, data["train_y"], vf, data["val_y"]),
            "linear": linear_probe(tf, data["train_y"], vf, data["val_y"])}


@torch.no_grad()
def predict_probs(model: nn.Module, x8: torch.Tensor, size: int, tta: bool = False, bs: int = 256) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(x8), bs):
        x = eval_view(x8[i:i + bs], size).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(x.flip(3)).float().softmax(1)) / 2
        out.append(p)
    return torch.cat(out)


def accuracy(probs: torch.Tensor, y: torch.Tensor) -> float:
    return (probs.argmax(1) == y).float().mean().item()
