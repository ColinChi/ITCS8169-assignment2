#!/usr/bin/env bash
# One-time test evaluation, run after the final recipe was locked on validation (ensemble A:
# DINO fine-tune x3 + SimCLR MixUp fine-tune x3, val 92.50%). Nothing was tuned after this.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/test
# Final system: models trained on all 2,400 images.
$PY evaluate.py runs/final_ft_dino224_s{0,1,2} runs/final_mix_simclr224_s{0,1,2} --split test --confirm-test --tta \
  --out results/test/final_ensemble.json
# Same ensemble built from the 1,920-image (validation-split) models.
$PY evaluate.py runs/ft_dino224_s{0,1,2} runs/mix_simclr224_s{0,1,2} --split test --confirm-test --tta \
  --out results/test/ensemble_A_1920.json
# Comparison rows (1,920-image models, single models, no TTA): mean over 3 seeds is computed in summarize.
for g in sup224 supmix224 ft_simclr224 mix_simclr224 ft_dino224 mix_dino224 sup224_r50 supmix224_r50; do
  $PY evaluate.py runs/${g}_s{0,1,2} --split test --confirm-test --out results/test/$g.json
done
