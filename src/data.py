"""Decode the dataset once into uint8 tensors that live on the GPU.

The dataset is tiny (2,800 images), so every image is resized to a fixed square
side and cached. All augmentation then runs batched on the GPU (``augment.py``),
which removes the CPU data-loading bottleneck.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "splits" / "seed0_val20.json"
CACHE_DIR = ROOT / "cache"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _decode(path: Path, side: int) -> np.ndarray:
    # Stretch to a square so the whole scene is kept; grayscale images become 3 identical channels.
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((side, side), Image.BICUBIC))


def build_cache(side: int) -> dict:
    manifest = json.loads(SPLIT_PATH.read_text())
    class_to_idx = manifest["class_to_idx"]
    test_paths = sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "data" / "test").glob("*/*")
        if p.suffix.lower() in IMG_EXTS
    )
    out = {"classes": sorted(class_to_idx, key=class_to_idx.get), "side": side}
    splits = {"train": manifest["train_paths"], "val": manifest["val_paths"], "test": test_paths}
    with ThreadPoolExecutor(32) as pool:
        for split, paths in splits.items():
            arrays = list(pool.map(lambda p: _decode(ROOT / p, side), paths))
            out[f"{split}_x"] = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
            out[f"{split}_y"] = torch.tensor([class_to_idx[Path(p).parent.name] for p in paths])
            out[f"{split}_paths"] = list(paths)
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = CACHE_DIR / f"data_{side}.pt.{os.getpid()}.tmp"  # unique: parallel runs may build it at once
    torch.save(out, tmp)
    tmp.rename(CACHE_DIR / f"data_{side}.pt")
    return out


def get_data(device: torch.device, side: int = 256, full_train: bool = False,
             label_frac: float = 1.0, seed: int = 0) -> dict:
    """Return GPU tensors. ``full_train`` merges the validation images into training (final model only).

    ``label_frac`` keeps a class-stratified subset of the training images (label-efficiency study).
    """
    path = CACHE_DIR / f"data_{side}.pt"
    data = torch.load(path, weights_only=False) if path.exists() else build_cache(side)
    if full_train:
        data["train_x"] = torch.cat([data["train_x"], data["val_x"]])
        data["train_y"] = torch.cat([data["train_y"], data["val_y"]])
    if label_frac < 1.0:
        g = torch.Generator().manual_seed(seed)
        keep = []
        for c in data["train_y"].unique():
            idx = (data["train_y"] == c).nonzero().flatten()
            n = max(1, round(label_frac * len(idx)))
            keep.append(idx[torch.randperm(len(idx), generator=g)[:n]])
        keep = torch.cat(keep).sort().values
        data["train_x"], data["train_y"] = data["train_x"][keep], data["train_y"][keep]
    for k in list(data):
        if k.endswith("_x") or k.endswith("_y"):
            data[k] = data[k].to(device)
    return data


def batches(n: int, bs: int, device: torch.device, drop_last: bool = True):
    """Yield random index batches covering one epoch."""
    perm = torch.randperm(n, device=device)
    stop = n - n % bs if drop_last else n
    for i in range(0, stop, bs):
        yield perm[i:i + bs]
