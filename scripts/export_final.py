"""Export the final models (trained on all 2,400 images) as compact FP16 inference checkpoints.

Usage: python scripts/export_final.py [run names...]   (default: the 224 px ensemble)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FINAL_RUNS = [f"final_ft_dino224_s{s}" for s in range(3)] + [f"final_mix_simclr224_s{s}" for s in range(3)]


def main():
    out = ROOT / "checkpoints"
    out.mkdir(exist_ok=True)
    lines = []
    for run in sys.argv[1:] or FINAL_RUNS:
        ckpt = torch.load(ROOT / "runs" / run / "model.pt", map_location="cpu", weights_only=False)
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in ckpt["model"].items()}
        path = out / f"{run}.pt"
        torch.save({"model": state, "config": ckpt["config"], "classes": ckpt["classes"]}, path)
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        print(path.name, f"{path.stat().st_size / 2**20:.1f} MiB")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
