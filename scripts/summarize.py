"""Aggregate runs/*/summary.json into results/ssl_probes.csv and results/finetune.csv (mean ± std over seeds)."""

from __future__ import annotations

import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ssl_rows, groups = [], defaultdict(list)
    for f in sorted((ROOT / "runs").glob("*/summary.json")):
        s = json.loads(f.read_text())
        name = f.parent.name
        if "method" in s:
            ssl_rows.append({"run": name, "method": s["method"], "backbone": s["backbone"], "full_train": s["full_train"],
                             "knn": s.get("knn", ""), "linear": s.get("linear", ""), "minutes": s["minutes"]})
        elif not s.get("full_train"):
            groups[re.sub(r"_s\d+$", "", name)].append(s)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "ssl_probes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["run", "method", "backbone", "full_train", "knn", "linear", "minutes"])
        w.writeheader()
        w.writerows(ssl_rows)
    fields = ["group", "n", "val", "val_std", "val_tta", "train_acc", "gap", "val_loss", "test", "test_std"]
    rows = []
    for g, ss in sorted(groups.items()):
        v = [s["val_acc"] for s in ss]
        rows.append({"group": g, "n": len(ss), "val": st.mean(v), "val_std": st.stdev(v) if len(v) > 1 else 0.0,
                     "val_tta": st.mean(s["val_acc_tta"] for s in ss),
                     "train_acc": st.mean(s["train_acc"] for s in ss), "gap": st.mean(s["gap"] for s in ss),
                     "val_loss": st.mean(s["val_loss"] for s in ss), "test": "", "test_std": ""})
        test = out / "test" / f"{g}.json"  # written once by scripts/test_eval.sh after the recipe was locked
        if test.exists():
            t = list(json.loads(test.read_text())["models"].values())
            rows[-1]["test"], rows[-1]["test_std"] = st.mean(t), (st.stdev(t) if len(t) > 1 else 0.0)
    with open(out / "finetune.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{'run':24s} {'knn':>6s} {'linear':>7s}")
    for r in ssl_rows:
        fmt = lambda x: f"{100 * x:6.1f}" if x != "" else "     -"  # noqa: E731
        print(f"{r['run']:24s} {fmt(r['knn'])} {fmt(r['linear'])}  ({r['minutes']} min)")
    print(f"\n{'group':24s} {'n':>2s} {'val':>12s} {'val+TTA':>8s} {'gap':>6s} {'val_loss':>8s} {'test':>12s}")
    for r in rows:
        print(f"{r['group']:24s} {r['n']:2d} {100 * r['val']:6.2f} ± {100 * r['val_std']:3.1f} "
              f"{100 * r['val_tta']:8.2f} {100 * r['gap']:6.1f} {r['val_loss']:8.3f} "
              + (f"{100 * r['test']:6.2f} ± {100 * r['test_std']:3.1f}" if r["test"] != "" else ""))


if __name__ == "__main__":
    main()
