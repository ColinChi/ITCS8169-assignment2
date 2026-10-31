# ITCS 6169/8169 Assignment 2 — Self-Supervised Representation Learning

Every network in this repository is **randomly initialized** (`torchvision.models.resnet18/resnet50(weights=None)`) and trained **only on the 2,400 provided training images**. No external weights, external images, or test images are used for pretraining, training, or model selection.

## Final result

| System | Validation (1,920/480 split) | **Test (400 images)** |
|---|---:|---:|
| Ensemble of 6 ResNet-18s at 224 px: 3 × DINO-pretrained → fine-tuned, 3 × SimCLR-pretrained → fine-tuned with MixUp/CutMix; SSL and fine-tuning on all 2,400 images; flip TTA | 92.5% (same configuration trained on 1,920 images) | **90.75%** (363/400) |

The configuration was selected on validation among three ensembles that were specified in advance (`results/val_ensemble_{A,B,C}.json`), before the test set was used. The six FP16 inference checkpoints (21 MiB each, SHA-256 in `checkpoints/SHA256SUMS`) are in [`checkpoints/`](checkpoints/). Evaluate them with:

```bash
python evaluate.py checkpoints/final_*.pt --split test --confirm-test --tta
```

The report is [`report/assignment2_report.pdf`](report/assignment2_report.pdf) (Word version: `report/assignment2_report.docx`).

## Setup

```bash
python -m pip install -r requirements.txt
```

Experiments used Python 3.10, PyTorch 2.14.1 + torchvision 0.29.1 (CUDA 13.0) on NVIDIA H200 GPUs shared with other jobs. Place the course dataset as

```text
data/train/<class>/*.jpg     # 2,400 images, 16 classes
data/test/<class>/*.jpg      # 400 images
```

The first run decodes every image once into `cache/data_<side>.pt` (uint8, side × side). All augmentation then runs batched on the GPU (`src/augment.py`).

## Protocol

- **Split.** `splits/seed0_val20.json`: class-stratified 1,920 train / 480 validation, exact-duplicate images kept on one side. All SSL pretraining for comparisons uses only the 1,920 training-split images.
- **Backbone.** ResNet-18 at 224×224 (images stored at 256×256) unless stated otherwise; ResNet-50 and 128 px are controls.
- **Fixed budgets, no early stopping.** SSL pretraining: 800 epochs. Every classifier: 200 epochs, SGD (Nesterov, lr 0.1, cosine, 5 warm-up epochs), batch 128, weight decay 5e-4, label smoothing 0.1; the **final-epoch** model is reported. The same recipe fine-tunes every SSL encoder, which makes retraining on all 2,400 images (`--full-train`) well defined.
- **Representation probes.** Weighted cosine kNN (k = 20) and a linear logistic-regression probe on frozen, standardized features.
- **Test set.** Touched only through `evaluate.py --confirm-test`, after the final configuration was fixed (`scripts/test_eval.sh`). Analysis-only rows that finished later were evaluated with `scripts/test_eval_analysis.sh` and played no role in selection.

## Code map

| File | Purpose |
|---|---|
| `src/data.py` | Decodes images once into GPU uint8 tensors; split handling; label-fraction subsets; epoch sampler |
| `src/augment.py` | Per-sample GPU random-resized-crop + flip (one affine warp), color jitter, grayscale, blur, solarize |
| `src/models.py` | Randomly initialized torchvision ResNet backbones and the linear classifier |
| `src/ssl.py` | **DINO** (multi-crop self-distillation, EMA teacher, centering) and **SimCLR** (NT-Xent) |
| `src/evaluation.py` | Feature extraction, kNN and linear probes, prediction with optional flip-TTA |
| `pretrain.py` | Self-supervised pretraining (labels are used only by the monitoring probes, never in the loss) |
| `finetune.py` | Supervised baseline, SSL fine-tuning, joint CE + λ·SimCLR, MixUp/CutMix, EMA, label fractions |
| `evaluate.py` | Accuracy of single models / probability-averaged ensembles on val or (guarded) test; accepts run directories or exported checkpoints |
| `scripts/*.sh` | Exact commands for every experiment (see below) |
| `scripts/summarize.py`, `scripts/build_report.py`, `scripts/export_final.py` | Result tables, the two-page PDF, FP16 export |

## Reproducing the experiments

Each script launches its runs in parallel on GPUs 0–3 (edit the ids if needed). Every run writes `runs/<name>/{config.json, log.jsonl, summary.json}` plus weights (`backbone.pt` or `model.pt`; weights are not committed except the final checkpoints).

| Order | Script | Experiments |
|---|---|---|
| 1 | `scripts/run_224.sh` | ResNet-18 at 224 px: supervised baseline (± MixUp/CutMix), DINO and SimCLR pretraining → fine-tuning (± MixUp/CutMix), 3 seeds each; ResNet-50 supervised and DINO; DINO and SimCLR pretraining on all 2,400 images |
| 2 | `scripts/final_ft_simclr.sh`, `scripts/final_ft_dino.sh` | Fine-tuning of the 2,400-image SSL backbones on all 2,400 images (final-model candidates) |
| 3 | `scripts/run_ablation.sh` | Joint CE + 0.5·SimCLR at 224 px; supervised / DINO / SimCLR at 128 px |
| 4 | `scripts/test_eval.sh`, `scripts/export_final.py` | One-time test evaluation of the selected system and the comparison rows; FP16 export |
| 5 | `scripts/test_eval_analysis.sh` | Test accuracy of analysis-only rows (joint, 128 px, ResNet-50 DINO, 256 px) |
| 6 | `scripts/run_256.sh` | The full pipeline at 256 px (native size of half of the images), including SSL pretraining and 2,400-image candidates |

Then `python scripts/summarize.py` writes `results/ssl_probes.csv` and `results/finetune.csv`, and `python scripts/build_report.py` builds the PDF and the Word version from the same text. (The 128 px part of `run_ablation.sh` was relaunched once after three processes raced to build the same image cache; `src/data.py` now writes the cache atomically per process.)

Wall-clock on a shared H200: DINO ResNet-18 pretraining 24 min, SimCLR 10 min, DINO ResNet-50 68 min, one 200-epoch fine-tuning run 3 min.

## Results summary

ResNet-18 unless noted; mean ± std over 3 fine-tuning seeds; single models without TTA. kNN / Linear = frozen encoder on validation. Test = the same models, evaluated once after the final configuration was fixed.

| Method | kNN | Linear | Val | Test |
|---|---:|---:|---:|---:|
| Supervised baseline, 224 px | – | – | 85.0 ± 0.4 | 83.7 ± 0.4 |
| Supervised + MixUp/CutMix | – | – | 87.9 ± 0.9 | 83.7 ± 1.4 |
| Joint CE + 0.5·SimCLR (from scratch) | – | – | 86.6 ± 0.6 | 82.6 ± 1.0 |
| SimCLR → fine-tune | 80.6 | 82.5 | 88.8 ± 0.8 | 85.2 ± 0.7 |
| SimCLR → fine-tune + MixUp/CutMix | – | – | 89.8 ± 1.0 | 87.8 ± 0.1 |
| **DINO → fine-tune** | **82.9** | **83.1** | **90.8 ± 0.7** | 88.0 ± 0.5 |
| DINO → fine-tune + MixUp/CutMix | – | – | 89.4 ± 1.3 | 88.3 ± 0.3 |
| 128 px: supervised / SimCLR → FT / DINO → FT | – / 80.4 / 81.5 | – / 78.8 / 83.5 | 81.0 / 85.5 / 87.4 | 80.1 / 80.9 / 84.3 |
| ResNet-50: supervised + MixUp / DINO → FT + MixUp | – / 81.0 | – / 82.7 | 86.9 / 90.1 | 84.7 / 89.1 |
| 256 px: supervised / SimCLR → FT / DINO → FT | – / 80.6 / 79.6 | – / 84.2 / 84.0 | 84.6 / 89.9 / 90.0 | 84.5 / 86.0 / 89.3 |
| Final ensemble (6 models, TTA), trained on 1,920 images | | | 92.5 | 88.5 |
| **Final ensemble, trained on all 2,400 images** | | | – | **90.75** |

**256 px study.** It was run after the 224 px system had been evaluated on test and followed the same validation rule (`results/val_ensemble_256_{A,B,C}.json`). Its best ensemble (B: 3 × DINO → FT + MixUp, 3 × SimCLR → FT + MixUp) reached 91.67% validation accuracy, below the 224 px system's 92.50%. The submitted system is therefore unchanged, and the 256 px ensembles trained on all 2,400 images were not evaluated on test. Single 256 px models score 0.5–1.5 points higher than their 224 px counterparts on test, but not consistently on validation.

Full tables: `results/finetune.csv`, `results/ssl_probes.csv`, `results/test/*.json`. See the report for the analysis and [`AI_USAGE.md`](AI_USAGE.md) for how AI tools were used.
