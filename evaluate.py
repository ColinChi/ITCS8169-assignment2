"""Evaluate fine-tuned models (and their probability-averaged ensemble) on validation or test.

The test split is guarded: it requires ``--confirm-test`` and should be used only
after the recipe has been locked on validation.

Example:
    python evaluate.py runs/ft_dino224_s0 runs/sup224_s0 --split val --tta
    python evaluate.py checkpoints/final_*.pt --split test --confirm-test --tta   # shipped final ensemble
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import get_data
from src.evaluation import accuracy, predict_probs
from src.models import Classifier, build_backbone

ROOT = Path(__file__).resolve().parent


def load(run: Path, dev: torch.device):
    """``run`` is a run directory (runs/<name>) or an exported checkpoint file (checkpoints/<name>.pt)."""
    run = Path(run)
    ckpt = torch.load(run if run.suffix == ".pt" else run / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = Classifier(build_backbone(cfg["backbone"]), num_classes=len(ckpt["classes"]))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()})
    return model.to(dev).to(memory_format=torch.channels_last).eval(), cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--confirm-test", action="store_true")
    p.add_argument("--tta", action="store_true", help="average with the horizontally flipped image")
    p.add_argument("--out", default=None, help="write results JSON here")
    a = p.parse_args()
    if a.split == "test" and not a.confirm_test:
        raise SystemExit("Refusing to touch the test set without --confirm-test")
    dev = torch.device("cuda")
    caches, probs, res = {}, [], {"split": a.split, "tta": a.tta, "models": {}}
    for r in a.runs:
        model, cfg = load(Path(r), dev)
        side = cfg.get("cache_size", 256)
        if side not in caches:
            caches[side] = get_data(dev, side=side)
        x, y = caches[side][f"{a.split}_x"], caches[side][f"{a.split}_y"]
        pr = predict_probs(model, x, cfg["size"], tta=a.tta)
        probs.append(pr)
        res["models"][r] = accuracy(pr, y)
        print(f"{r}: {100 * res['models'][r]:.2f}%")
    if len(probs) > 1:
        res["ensemble"] = accuracy(torch.stack(probs).mean(0), y)
        print(f"ensemble of {len(probs)}: {100 * res['ensemble']:.2f}%")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
