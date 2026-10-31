#!/usr/bin/env bash
# Test evaluation of analysis-only rows (joint training, 128 px, ResNet-50 DINO). These runs finished after the
# final configuration had been fixed (scripts/test_eval.sh) and play no role in model selection.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
for g in joint224 sup128 supmix128 ft_simclr128 mix_simclr128 ft_dino128 mix_dino128 ft_dino224_r50 mix_dino224_r50; do
  $PY evaluate.py runs/${g}_s{0,1,2} --split test --confirm-test --out results/test/$g.json | tail -1
done
# 256 px study (run after the 224 px system had been evaluated on test). Its best validation ensemble (B, 91.67%)
# did not exceed the 224 px system (92.50%), so the submitted system is unchanged and the 256 px ensembles trained
# on all 2,400 images are not evaluated on test. Single-model rows are evaluated for analysis only.
for g in sup256 supmix256 joint256 ft_simclr256 mix_simclr256 ft_dino256 mix_dino256; do
  $PY evaluate.py runs/${g}_s{0,1,2} --split test --confirm-test --out results/test/$g.json | tail -1
done
