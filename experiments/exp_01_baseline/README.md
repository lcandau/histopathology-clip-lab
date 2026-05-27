# exp_01_baseline — CLIP with ResNet50 + BERT on LC25000

CLIP-style image-text contrastive learning with frozen general-purpose backbones, evaluated on the Kaggle LC25000 dataset. This folder hosts the *CLIP + ResNet50 family*: the controlled baseline and its partial-fine-tuning ablation.

Related folders:
- `experiments/exp_02_plip_zeroshot/` — pathology-pretrained CLIP (PLIP) zero-shot reference point.
- `experiments/exp_03_plain_classifier/` — no-CLIP-architecture control (ResNet50 + Dense(5) + softmax).

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `CLIP_ResNet50_baseline.ipynb` | Current baseline. Frozen ResNet50 (ImageNet) + frozen BERT base, projection heads + learnable temperature, symmetric InfoNCE. |
| `CLIP_ResNet50_unfreeze30.ipynb` | Partial fine-tuning ablation. Same recipe as the baseline but the last 30 layers of ResNet50 are trainable; LR dropped 10x. |
| `CLIP_ResNet50_v1.ipynb` ... `v6.ipynb` | Earlier proof-of-concept iterations. Kept for reference; not part of the project's experiment set. |

## Archived runs

Completed runs live under `runs/<YYYY-MM-DD>_<tag>/` with the executed notebook (outputs embedded).

| Date | Tag | Notebook | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| 2026-05-24 | `baseline` (initial) | `CLIP_ResNet50_baseline.ipynb` | 0.6300 | 0.6247 |
| 2026-05-25 | `baseline` (rerun, canonical) | `CLIP_ResNet50_baseline.ipynb` | 0.6356 | 0.6273 |
| 2026-05-25 | `unfreeze30` | `CLIP_ResNet50_unfreeze30.ipynb` | 0.8984 | 0.8981 |

## What this baseline isolates

- Frozen backbones $\to$ no feature-level adaptation.
- Single fixed prompt template at training and evaluation.
- No augmentation (LC25000 is already pre-augmented from 1,250 originals).
- Stratified 80/10/10 split (seed 42) shared across all experiments.

Every other notebook in this directory changes exactly one of those variables relative to the baseline, so the resulting deltas are attributable.
