"""Supervised training: from scratch (baseline) or from an SSL backbone (fine-tuning / linear probe).

Fixed budget, no early stopping: the final-epoch model is reported, so retraining
on all 2,400 images (``--full-train``) is well defined.

Example:
    python finetune.py --name sup224_s0 --seed 0
    python finetune.py --name ft_dino224_s0 --init runs/dino224/backbone.pt --seed 0
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from src.augment import train_view
from src.data import batches, get_data
from src.evaluation import accuracy, predict_probs
from src.models import Classifier, build_backbone
from src.ssl import SimCLR
from src.utils import RunLogger, cosine_lr, seed_all, set_lr


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--init", default=None, help="backbone.pt from pretrain.py")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--crop-min", type=float, default=0.35)
    p.add_argument("--mixup", type=float, default=0.0, help="MixUp/CutMix Beta alpha; 0 disables")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights and BN stats; 0 disables")
    p.add_argument("--joint-weight", type=float, default=0.0,
                   help="joint training: add this weight x SimCLR loss on the same backbone; 0 disables")
    p.add_argument("--freeze", action="store_true", help="train only the linear head (linear probe with augmentation)")
    p.add_argument("--label-frac", type=float, default=1.0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--cache-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full-train", action="store_true")
    p.add_argument("--eval-every", type=int, default=10)
    return p.parse_args()


def mix(x, y, alpha, rng):
    """MixUp or CutMix (50/50) on a normalized batch; returns inputs, both targets, and the mixing weight."""
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(len(x), device=x.device)
    if rng.random() < 0.5:
        return lam * x + (1 - lam) * x[perm], y, y[perm], lam
    h, w = x.shape[-2:]
    ch, cw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
    cy, cx = rng.integers(h), rng.integers(w)
    y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[..., y0:y1, x0:x1] = x[perm][..., y0:y1, x0:x1]
    return x, y, y[perm], 1 - (y1 - y0) * (x1 - x0) / (h * w)


def main():
    a = parse()
    seed_all(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    data = get_data(dev, side=a.cache_size, full_train=a.full_train, label_frac=a.label_frac, seed=a.seed)
    backbone = build_backbone(a.backbone)
    if a.init:
        backbone.load_state_dict(torch.load(a.init, map_location="cpu"))
    model = Classifier(backbone, num_classes=len(data["classes"]), dropout=a.dropout).to(dev)
    model = model.to(memory_format=torch.channels_last)
    if a.freeze:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    joint = None
    if a.joint_weight > 0:  # SimCLR shares model.backbone; only its projection head is extra
        joint = SimCLR(model.backbone, size=a.size, crop_min=0.08).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if joint is not None:
        params += list(joint.proj.parameters())
    opt = torch.optim.SGD([{"params": [p for p in params if p.ndim > 1], "weight_decay": a.wd},
                           {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)
    ema = None
    if a.ema > 0:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    log = RunLogger(a.name, vars(a))

    x8, y = data["train_x"], data["train_y"]
    n = len(x8)
    bs = min(a.bs, n)
    steps_per_epoch = n // bs
    total, step = a.epochs * steps_per_epoch, 0
    for epoch in range(a.epochs):
        model.train()
        if a.freeze:
            model.backbone.eval()  # keep BN statistics of the SSL encoder fixed
        loss_sum, correct = 0.0, 0
        for idx in batches(n, bs, dev):
            set_lr(opt, cosine_lr(step, total, a.warmup_epochs * steps_per_epoch, a.lr))
            x = train_view(x8[idx], a.size, a.crop_min)
            ya = yb = y[idx]
            lam = 1.0
            if a.mixup > 0:
                x, ya, yb, lam = mix(x, ya, a.mixup, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x.contiguous(memory_format=torch.channels_last)).float()
            loss = lam * F.cross_entropy(out, ya, label_smoothing=a.label_smoothing) + \
                (1 - lam) * F.cross_entropy(out, yb, label_smoothing=a.label_smoothing)
            if joint is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss + a.joint_weight * joint(x8[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.is_floating_point():
                            ema[k].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
                        else:
                            ema[k].copy_(v)
            loss_sum += loss.item()
            correct += (out.argmax(1) == ya).sum().item()
            step += 1
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "loss": loss_sum / steps_per_epoch,
               "train_acc_aug": correct / (steps_per_epoch * bs)}
        if (epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs:
            row["val_acc"] = accuracy(predict_probs(model, data["val_x"], a.size), data["val_y"])
        log.write(**row)

    summary = {"val_acc_raw": row["val_acc"]}
    if ema is not None:
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()})
    probs = predict_probs(model, data["val_x"], a.size)
    summary["val_acc"] = accuracy(probs, data["val_y"])
    summary["val_acc_tta"] = accuracy(predict_probs(model, data["val_x"], a.size, tta=True), data["val_y"])
    summary["val_loss"] = F.nll_loss(probs.clamp_min(1e-8).log(), data["val_y"]).item()
    summary["train_acc"] = accuracy(predict_probs(model, x8, a.size), y)
    summary["gap"] = summary["train_acc"] - summary["val_acc"]
    summary["full_train"] = a.full_train  # if True, val images were trained on and val_acc is not a held-out number
    torch.save({"model": model.state_dict(), "config": vars(a), "classes": data["classes"]}, log.dir / "model.pt")
    log.finish(**summary)


if __name__ == "__main__":
    main()
