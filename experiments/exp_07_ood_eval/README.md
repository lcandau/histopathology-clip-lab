# exp_07 — Clean OOD evaluation (LC25000-trained → NCT-CRC-HE-7K)

Fresh OOD evaluation built on top of the corrected baseline:

- **Preprocessing**: `/255.0` via `src.data.images.load_image` (matches keras_hub ResNet50)
- **Training set**: LC25000 deduped (23,720 files; 1,280 byte-duplicates removed)
- **Checkpoint loaded**: `clip_histopathology/exp_01_baseline/baseline/weights.weights.h5` from Drive — the run produced by the patched `exp_01_baseline/CLIP_ResNet50_baseline.ipynb` after the preprocessing + dedupe fix
- **OOD test set**: NCT-CRC-HE-7K (Kather 2019) restricted to the two LC25000-mapped classes (NORM ↔ benign colon, TUM ↔ colon adeno)

## Why a fresh notebook (and not extending exp_05)

`exp_05_ood_transfer` carries broken preprocessing (raw `[0, 255]` PIL floats fed into a backbone expecting `[0, 1]`), variant-juggling logic for stain experiments, and 37 cells of accreted complexity. exp_07 starts from a clean slate so the new comparison numbers can be trusted at face value.

## What this notebook produces

| Artifact | Path |
|---|---|
| Global metrics (acc, balanced acc, macro/weighted F1) | `results/metrics/exp_07_ood_eval/baseline_ood_classification.json` |
| Confusion matrix (raw + row-normalised) | `results/confusion_matrices/exp_07_ood_eval/baseline_ood_cm_*.npy` |
| OOD test-set UMAP (image embeddings + 2 colon prompt vectors) | `results/plots/exp_07_ood_eval/baseline_ood_umap.png` |

## What it does NOT do (deferred)

- LC25000 in-distribution evaluation (the training notebook already prints those numbers)
- PLIP zero-shot comparison (PLIP eval currently outputs 0.0000 — needs a separate bug fix)
- Stain normalisation variants (separate notebook)
- 9 × 5 NCT-leak confusion matrix (separate analysis)

Those come back when the comparison is ready, in follow-up notebooks (exp_07b, exp_08, …).
