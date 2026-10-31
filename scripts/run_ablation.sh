#!/usr/bin/env bash
# Controlled ablations with the same code and recipes as scripts/run_224.sh:
#  (1) joint training at 224 px: CE + 0.5 x SimCLR on one backbone from scratch (no pretraining), 3 seeds;
#  (2) resolution: the supervised / DINO / SimCLR pipeline at 128 px (cache 146 px, same 1.14 ratio as 256/224;
#      DINO local crops scaled 96 -> 56 px), 3 fine-tuning seeds each.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
pre() { gpu=$1; name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY pretrain.py --name $name "$@" > logs/$name.log 2>&1; }
ft()  { gpu=$1; name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY finetune.py --name $name "$@" > logs/$name.log 2>&1; }
R128="--size 128 --cache-size 146"
DINO="--method dino --bs 64 --lr 0.05 --wd 1e-4 --out-dim 1024 --teacher-temp 0.04"
SIMCLR="--method simclr --bs 256 --lr 0.3 --wd 1e-4 --warmup-epochs 20 --crop-min 0.08"
ft3() { gpu=$1; tag=$2; init=$3; shift 3
  for s in 0 1 2; do ft $gpu ft_${tag}_s$s --init $init --seed $s "$@" & done; wait
  for s in 0 1 2; do ft $gpu mix_${tag}_s$s --init $init --seed $s --mixup 0.4 "$@" & done; wait; }

( for s in 0 1 2; do ft 3 joint224_s$s --joint-weight 0.5 --seed $s & done; wait ) &
( pre 0 dino128 $DINO $R128 --local-size 56 && ft3 0 dino128 runs/dino128/backbone.pt $R128 ) &
( pre 1 simclr128 $SIMCLR $R128 && ft3 1 simclr128 runs/simclr128/backbone.pt $R128 ) &
( for s in 0 1 2; do ft 2 sup128_s$s --seed $s $R128 & done; wait
  for s in 0 1 2; do ft 2 supmix128_s$s --seed $s --mixup 0.4 $R128 & done; wait ) &
wait
echo "run_ablation done $(date)"
