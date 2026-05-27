# exp_02_plip_zeroshot — PLIP zero-shot reference point

Pathology-pretrained CLIP (`vinid/plip`, Huang 2023) evaluated zero-shot on the same LC25000 stratified 80/10/10 test split as `exp_01_baseline`. Isolates the contribution of pathology-domain pretraining relative to the general-purpose CLIP baseline.

Related folders:
- `experiments/exp_01_baseline/` — general-purpose CLIP baseline this is compared against.
- `experiments/exp_03_plain_classifier/` — no-CLIP-architecture control row.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `CLIP_PLIP_zeroshot.ipynb` | PLIP loaded from HuggingFace, evaluated zero-shot. No LC25000 training. Same prompts as the baseline. |

## Archived runs

| Date | Tag | Notebook | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| 2026-05-24 | `plip_zeroshot` (original) | `CLIP_PLIP_zeroshot.ipynb` | 0.6980 | 0.7004 |
| 2026-05-26 | `plip_zeroshot` (rerun with shared-UMAP helper) | `CLIP_PLIP_zeroshot.ipynb` | 0.6980 | 0.7004 |

## What this reference point isolates

- Same evaluation protocol as the baseline (argmax of cosine between test image embedding and the five class-prompt embeddings).
- No LC25000-specific training. The model only sees the test images at inference time.
- The $\Delta$ vs `exp_01_baseline` measures pathology-domain pretraining within the frozen-encoder regime.

Headline finding (vs the canonical 2026-05-25 baseline rerun): $+7.3$ macro-F1. Gain concentrates on benign tissues; the two cancer subtypes essentially tie.
