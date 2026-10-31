"""Self-supervised pretraining on the training images only (labels are never used in the loss).

Labels are used only by the periodic kNN / linear probes that monitor representation quality.

Example:
    python pretrain.py --method dino --name dino224 --epochs 800
"""

from __future__ import annotations

import argparse
import math

import torch

from src.data import batches, get_data
from src.evaluation import probes
from src.models import build_backbone
from src.ssl import DINO, SimCLR
from src.utils import RunLogger, cosine_lr, seed_all, set_lr


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["dino", "simclr"])
    p.add_argument("--name", required=True)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05, help="SGD lr")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--clip", type=float, default=3.0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--cache-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full-train", action="store_true", help="also use the val images, unlabeled (final model only)")
    p.add_argument("--eval-every", type=int, default=50)
    # DINO
    p.add_argument("--out-dim", type=int, default=1024)
    p.add_argument("--n-local", type=int, default=6)
    p.add_argument("--local-size", type=int, default=96)
    p.add_argument("--global-crop-min", type=float, default=0.25)
    p.add_argument("--teacher-temp", type=float, default=0.04)
    p.add_argument("--teacher-momentum", type=float, default=0.996)
    p.add_argument("--freeze-last-epochs", type=int, default=1)
    # SimCLR
    p.add_argument("--crop-min", type=float, default=0.08)
    p.add_argument("--temperature", type=float, default=0.2)
    return p.parse_args()


def param_groups(model: torch.nn.Module, wd: float) -> list[dict]:
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def main():
    a = parse()
    seed_all(a.seed)
    dev = torch.device("cuda")
    data = get_data(dev, side=a.cache_size, full_train=a.full_train)
    backbone = build_backbone(a.backbone)
    if a.method == "dino":
        model = DINO(backbone, out_dim=a.out_dim, n_local=a.n_local, global_size=a.size, local_size=a.local_size,
                     global_scale=(a.global_crop_min, 1.0), teacher_temp=a.teacher_temp)
    else:
        model = SimCLR(backbone, size=a.size, crop_min=a.crop_min, temperature=a.temperature)
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(param_groups(model, a.wd), lr=a.lr, momentum=0.9)
    log = RunLogger(a.name, vars(a))

    n = len(data["train_x"])
    steps_per_epoch = n // a.bs
    total = a.epochs * steps_per_epoch
    step = 0
    for epoch in range(a.epochs):
        model.train()
        running = 0.0
        for idx in batches(n, a.bs, dev):
            set_lr(opt, cosine_lr(step, total, a.warmup_epochs * steps_per_epoch, a.lr))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(data["train_x"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            if a.method == "dino" and epoch < a.freeze_last_epochs:
                model.cancel_last_layer_grad()
            opt.step()
            if a.method == "dino":
                m = 1 - (1 - a.teacher_momentum) * (math.cos(math.pi * step / total) + 1) / 2
                model.update_teacher(m)
            running += loss.item()
            step += 1
        row = {"epoch": epoch + 1, "loss": running / steps_per_epoch, "lr": opt.param_groups[0]["lr"]}
        if not math.isfinite(row["loss"]):
            raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
        if not a.full_train and ((epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs):
            row.update(probes(model.encoder(), data, a.size))
        log.write(**row)

    torch.save(model.encoder().state_dict(), log.dir / "backbone.pt")
    final = {k: v for k, v in row.items() if k in ("knn", "linear", "loss")}
    log.finish(method=a.method, backbone=a.backbone, epochs=a.epochs, full_train=a.full_train, **final)


if __name__ == "__main__":
    main()
