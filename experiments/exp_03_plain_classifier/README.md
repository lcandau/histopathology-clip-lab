# exp_03_plain_classifier — ResNet50 + Dense(5) + softmax

No-CLIP-architecture control row for the project. Same ResNet50 backbone and the same LC25000 stratified 80/10/10 split as `exp_01_baseline`, but the entire CLIP paradigm is removed: no text encoder, no projection head, no learnable temperature, no contrastive loss. A single `Dense(5)` classifier sits on top of the global-average-pooled ResNet50 features and is trained with sparse cross-entropy. The $\Delta$ against the CLIP variants in `exp_01_baseline` isolates what the CLIP architecture contributes once the image encoder and data are held constant.

Related folders:
- `experiments/exp_01_baseline/` — CLIP variants on the same backbone (head-to-head comparison target).
- `experiments/exp_02_plip_zeroshot/` — pathology-pretrained CLIP reference point.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `ResNet50_plain_classifier.ipynb` | Frozen ResNet50 + BatchNorm + `Dense(5)`, CE loss, LR 1e-3. Pairs with `CLIP_ResNet50_baseline.ipynb`. |
| `ResNet50_plain_classifier_unfreeze30.ipynb` | Same recipe but with the last 30 ResNet50 layers unfrozen, LR 1e-4. Pairs with `CLIP_ResNet50_unfreeze30.ipynb`. |

The two variants fill the 2×2 head-to-head matrix CLIP-vs-CE at each feature-adaptation budget (frozen vs partial fine-tune).

## Archived runs

| Date | Tag | Notebook | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| 2026-05-25 | `plain_classifier` | `ResNet50_plain_classifier.ipynb` | 0.6440 | 0.6369 |
| 2026-05-26 | `plain_classifier_unfreeze30` | `ResNet50_plain_classifier_unfreeze30.ipynb` | 0.8728 | 0.8723 |

## What this experiment isolates

- Same frozen ResNet50 features as `exp_01_baseline`, but the head and loss differ.
- No prompts, no text encoder, no BERT — pure image-only classification.
- The $\Delta$ vs the CLIP variants measures the *cost or benefit of the CLIP architecture* on a closed 5-class task.
