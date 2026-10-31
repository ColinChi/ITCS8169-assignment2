"""Self-supervised objectives: DINO (self-distillation) and SimCLR (contrastive).

Both take a uint8 image batch, build their own augmented views on the GPU, and
return a scalar loss. Labels are never seen.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import ssl_view


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 3) -> nn.Sequential:
    mods, d = [], in_dim
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.GELU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class DINOHead(nn.Module):
    """MLP -> L2-normalized bottleneck -> weight-normalized prototypes (unit norm, as norm_last_layer=True)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048, bottleneck: int = 256) -> None:
        super().__init__()
        self.mlp = _mlp(in_dim, hidden, bottleneck)
        self.prototypes = nn.Parameter(torch.empty(out_dim, bottleneck))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1)
        return F.linear(x, F.normalize(self.prototypes, dim=-1))


class DINO(nn.Module):
    """Student/teacher self-distillation with multi-crop.

    The teacher is an exponential moving average of the student. It sees only
    the two global crops; the student must match the teacher's (centered,
    sharpened) distribution over prototypes from every other crop, including
    small local crops. Centering + sharpening prevent collapse without negatives.
    """

    def __init__(self, backbone: nn.Module, out_dim: int = 1024, n_local: int = 6, global_size: int = 224,
                 local_size: int = 96, global_scale=(0.25, 1.0), local_scale=(0.05, 0.25),
                 student_temp: float = 0.1, teacher_temp: float = 0.04, center_momentum: float = 0.9) -> None:
        super().__init__()
        self.student = nn.Sequential(backbone, DINOHead(backbone.out_dim, out_dim))
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.n_local, self.global_size, self.local_size = n_local, global_size, local_size
        self.global_scale, self.local_scale = global_scale, local_scale
        self.student_temp, self.teacher_temp, self.center_momentum = student_temp, teacher_temp, center_momentum

    def encoder(self) -> nn.Module:
        return self.teacher[0]  # the teacher backbone is the better, smoother encoder

    def forward(self, x8: torch.Tensor) -> torch.Tensor:
        gs, gsc = self.global_size, self.global_scale
        g = torch.cat([ssl_view(x8, gs, gsc, blur_p=1.0), ssl_view(x8, gs, gsc, blur_p=0.1, solarize_p=0.2)])
        with torch.no_grad():
            t_out = self.teacher(g).float()
            t_prob = F.softmax((t_out - self.center) / self.teacher_temp, dim=-1).chunk(2)
        s_out = list(self.student(g).float().chunk(2))
        if self.n_local:
            loc = torch.cat([ssl_view(x8, self.local_size, self.local_scale, blur_p=0.5) for _ in range(self.n_local)])
            s_out += list(self.student(loc).float().chunk(self.n_local))
        total, n = 0.0, 0
        for iq, q in enumerate(t_prob):
            for iv, s in enumerate(s_out):
                if iv == iq:
                    continue
                total = total + torch.sum(-q * F.log_softmax(s / self.student_temp, dim=-1), dim=-1).mean()
                n += 1
        with torch.no_grad():
            self.center.mul_(self.center_momentum).add_(t_out.mean(0, keepdim=True), alpha=1 - self.center_momentum)
        return total / n

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)

    def cancel_last_layer_grad(self) -> None:
        self.student[1].prototypes.grad = None


class SimCLR(nn.Module):
    """Two augmented views of each image; NT-Xent pulls them together and pushes away all other images."""

    def __init__(self, backbone: nn.Module, size: int = 224, crop_min: float = 0.08, temperature: float = 0.2,
                 proj_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.proj = _mlp(backbone.out_dim, 2048, proj_dim, layers=2)
        self.size, self.scale, self.temperature = size, (crop_min, 1.0), temperature

    def encoder(self) -> nn.Module:
        return self.backbone

    def forward(self, x8: torch.Tensor) -> torch.Tensor:
        v = torch.cat([ssl_view(x8, self.size, self.scale), ssl_view(x8, self.size, self.scale)])
        z = F.normalize(self.proj(self.backbone(v)).float(), dim=-1)
        n = x8.shape[0]
        sim = z @ z.t() / self.temperature
        sim.fill_diagonal_(-math.inf)
        target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, target)
