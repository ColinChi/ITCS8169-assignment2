#!/usr/bin/env bash
# Final-model candidates: fine-tune the all-2,400-image SimCLR backbone on all 2,400 labeled images
# (same fixed recipe as the validation runs; selection is made on validation results only).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPU=${GPU:-1}
ft() { name=$1; shift; CUDA_VISIBLE_DEVICES=$GPU $PY finetune.py --name $name --full-train "$@" > logs/$name.log 2>&1; }
for s in 0 1 2; do ft final_ft_simclr224_s$s --init runs/simclr224_full/backbone.pt --seed $s & done; wait
for s in 0 1 2; do ft final_mix_simclr224_s$s --init runs/simclr224_full/backbone.pt --seed $s --mixup 0.4 & done; wait
