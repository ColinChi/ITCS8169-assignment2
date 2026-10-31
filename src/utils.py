"""Run directories, logging, seeding, and learning-rate schedules."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def cosine_lr(step: int, total: int, warmup: int, base: float, final: float = 0.0) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return final + (base - final) * 0.5 * (1 + math.cos(math.pi * t))


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)


class RunLogger:
    def __init__(self, name: str, config: dict) -> None:
        self.dir = ROOT / "runs" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))
        self.log = open(self.dir / "log.jsonl", "w")
        self.t0 = time.time()

    def write(self, **row) -> None:
        row = {"time": round(time.time() - self.t0, 1), **row}
        self.log.write(json.dumps(row) + "\n")
        self.log.flush()
        print(json.dumps(row), flush=True)

    def finish(self, **summary) -> None:
        summary["minutes"] = round((time.time() - self.t0) / 60, 2)
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print("SUMMARY", json.dumps(summary), flush=True)
