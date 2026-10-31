#!/usr/bin/env bash
# 224 px study on 4 GPUs. Every classifier: 200 epochs, SGD lr 0.1, cosine, label smoothing 0.1,
# final-epoch model (no early stopping). SSL: 800 epochs on the 1,920 training-split images (val/test unused).
# "*_full" runs use all 2,400 training images with the same fixed hyperparameters (final model candidates);
# they are trained speculatively in parallel, and selection uses validation results only.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
pre() { gpu=$1; name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY pretrain.py --name $name "$@" > logs/$name.log 2>&1; }
ft()  { gpu=$1; name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $PY finetune.py --name $name "$@" > logs/$name.log 2>&1; }
# fine-tune one backbone with 3 seeds, plain and with MixUp/CutMix (alpha 0.4)
ft3() { gpu=$1; tag=$2; init=$3; bb=$4
  for s in 0 1 2; do ft $gpu ft_${tag}_s$s --init $init --backbone $bb --seed $s & done; wait
  for s in 0 1 2; do ft $gpu mix_${tag}_s$s --init $init --backbone $bb --seed $s --mixup 0.4 & done; wait; }
DINO="--method dino --bs 64 --lr 0.05 --wd 1e-4 --out-dim 1024 --teacher-temp 0.04"
SIMCLR="--method simclr --bs 256 --lr 0.3 --wd 1e-4 --warmup-epochs 20 --crop-min 0.08"

# GPU 0: DINO ResNet-18 (val split) -> fine-tune. (A ResNet-50 DINO run on all 2,400 images was started here
# as well but stopped after a few epochs because it could not finish in time; it is not part of any result.)
( pre 0 dino224 $DINO && ft3 0 dino224 runs/dino224/backbone.pt resnet18 ) &
# GPU 1: final-model candidates on all 2,400 images
( pre 1 dino224_full $DINO --full-train ) &
( pre 1 simclr224_full $SIMCLR --full-train ) &
# GPU 2: supervised baselines from scratch (ResNet-18), then SimCLR (val split) -> fine-tune
( for s in 0 1 2; do ft 2 sup224_s$s --seed $s & done; wait
  for s in 0 1 2; do ft 2 supmix224_s$s --seed $s --mixup 0.4 & done; wait
  pre 2 simclr224 $SIMCLR && ft3 2 simclr224 runs/simclr224/backbone.pt resnet18 ) &
# GPU 3: backbone study: DINO ResNet-50 (val split) -> fine-tune, and ResNet-50 supervised baselines
( pre 3 dino224_r50 $DINO --backbone resnet50 && ft3 3 dino224_r50 runs/dino224_r50/backbone.pt resnet50 ) &
( for s in 0 1 2; do ft 3 sup224_r50_s$s --backbone resnet50 --seed $s; done
  for s in 0 1 2; do ft 3 supmix224_r50_s$s --backbone resnet50 --seed $s --mixup 0.4; done ) &
wait
echo "run_224 done $(date)"
