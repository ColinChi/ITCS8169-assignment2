#!/usr/bin/env bash
# Main study at 256 px, the native size of half of the images (cache 256 x 256, no resizing at evaluation).
# Same recipes as scripts/run_224.sh; DINO local crops scaled 96 -> 112 px. SSL: 800 epochs.
#   - 1,920-image split: supervised (+/- MixUp/CutMix), joint CE + 0.5 SimCLR, DINO / SimCLR -> fine-tune (+/- MixUp)
#   - all 2,400 images: DINO / SimCLR pretraining and the fine-tuned final-model candidates (trained in parallel to
#     save time; selection uses validation only, with the same three pre-specified ensembles as at 224 px).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
R="--size 256 --cache-size 256"
pre() { gpu=$1; name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY pretrain.py --name $name $R "$@" > logs/$name.log 2>&1; }
ft()  { gpu=$1; name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY finetune.py --name $name $R "$@" > logs/$name.log 2>&1; }
# fine-tune one backbone with 3 seeds, plain and with MixUp/CutMix; $4 = extra flags (e.g. --full-train), $5 = name prefix
ft3() { gpu=$1; tag=$2; init=$3; extra=${4:-}; pfx=${5:-}
  for s in 0 1 2; do ft $gpu ${pfx}ft_${tag}_s$s --init $init --seed $s $extra & done; wait
  for s in 0 1 2; do ft $gpu ${pfx}mix_${tag}_s$s --init $init --seed $s --mixup 0.4 $extra & done; wait; }
DINO="--method dino --bs 64 --lr 0.05 --wd 1e-4 --out-dim 1024 --teacher-temp 0.04 --local-size 112"
SIMCLR="--method simclr --bs 256 --lr 0.3 --wd 1e-4 --warmup-epochs 20 --crop-min 0.08"

( pre 0 dino256 $DINO && ft3 0 dino256 runs/dino256/backbone.pt ) &
( pre 1 dino256_full $DINO --full-train && ft3 1 dino256 runs/dino256_full/backbone.pt --full-train final_ ) &
( for s in 0 1 2; do ft 2 sup256_s$s --seed $s & done; wait
  for s in 0 1 2; do ft 2 supmix256_s$s --seed $s --mixup 0.4 & done; wait
  pre 2 simclr256 $SIMCLR && ft3 2 simclr256 runs/simclr256/backbone.pt ) &
( for s in 0 1 2; do ft 3 joint256_s$s --joint-weight 0.5 --seed $s & done; wait
  pre 3 simclr256_full $SIMCLR --full-train && ft3 3 simclr256 runs/simclr256_full/backbone.pt --full-train final_ ) &
wait
echo "run_256 done $(date)"
