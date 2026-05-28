# Archived experiments

Experiments that produced useful artefacts during development but are no longer part of the active ablation suite. Kept here for traceability and to make pre-pivot results reachable from git history without cluttering the live `experiments/` listing.

## Why archived, not deleted

The thesis writeup cites results from these runs (especially the §limitations discussion of the keras_hub / Caffe-preprocessing bug and the LC25000 augmentation-leakage finding). Their notebooks are the record of how those numbers were produced. Deleting them would orphan the citations.

## What's here

### `exp_03b_unfreeze30/`
First-pass "unfreeze last 30 ResNet50 layers" CLIP training. Produced under the broken Caffe preprocessing; numbers (F1 ≈ 0.90 in-dist) are inflated/inconsistent vs the corrected pipeline. After the preprocessing fix the frozen baseline already saturates at ≈0.99 in-dist, so unfreezing the backbone is null on LC25000 — confirmed reason to drop it from the active ablation suite.

### `exp_05_ood_transfer/`
First OOD evaluation notebook (LC25000-trained → NCT-CRC-HE-7K). Superseded by `exp_07_ood_eval/` which:
- Uses the corrected `/255.0` preprocessing
- Uses the LC25000 deduped split
- Has cleaner UMAP visualisations (4 single-plot views + a 2×3 comparison grid)

Old `exp_05` numbers (e.g. baseline OOD F1 ≈ 0.21) reflect the broken Caffe preprocessing and should not be cited as the model's actual generalisation behaviour.

## Active ablation suite (going forward)

```
experiments/
  exp_01_baseline           — baseline CLIP, frozen ResNet50 + frozen BERT
  exp_02_plip_zeroshot      — PLIP zero-shot reference point
  exp_03_plain_classifier   — non-CLIP control: frozen ResNet50 + Dense(5) CE
  exp_04_stain_normalization — Macenko / Reinhard / Vahadane training-time normalisation
  exp_06_biomed_text        — text encoder ablation (BERT vs BioBERT vs PubMedBERT × name_only vs composed)
  exp_07_ood_eval           — cross-dataset evaluation: LC25000-trained → NCT-CRC
```
