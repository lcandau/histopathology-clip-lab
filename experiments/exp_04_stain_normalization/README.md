# exp_04_stain_normalization — stain-shortcut probe + Macenko/Reinhard/Vahadane head-to-head

Single-variable change from the baseline (`exp_01_baseline/CLIP_ResNet50_baseline.ipynb`): every image in the LC25000 split is passed through a deterministic stain-normalization step before reaching the model. The `STAIN_METHOD` knob in the notebook selects one of three methods per run; three Colab sessions in total build the head-to-head.

Related folders:
- `experiments/exp_01_baseline/` — unnormalized baseline this is compared against.

## Notebook

| Notebook | Purpose |
| --- | --- |
| `CLIP_ResNet50_stain.ipynb` | Stain-normalization ablation. Set `STAIN_METHOD ∈ {macenko, reinhard, vahadane}` in cell 3, Run All, archive the executed notebook + artifacts under `runs/<date>_stain_<method>/`. |

## Stain-normalization helpers

The three fitters live in `src/data/stain.py`:

- **`ReinhardFitter`** — match LAB-space mean/std to the reference.
- **`MacenkoFitter`** — SVD-based stain decomposition in optical-density space, percentile-based concentration normalisation.
- **`VahadaneFitter`** — sparse NMF (sklearn) for stain-matrix estimation, concentrations re-mapped to the reference's percentile.

All three are pure NumPy + scikit-image + scikit-learn. No `staintools` dependency (license-clean).

## Archived runs

| Date | Tag | Notebook | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| _pending_ | `stain_macenko` | `CLIP_ResNet50_stain.ipynb` | _pending_ | _pending_ |
| _pending_ | `stain_reinhard` | `CLIP_ResNet50_stain.ipynb` | _pending_ | _pending_ |
| _pending_ | `stain_vahadane` | `CLIP_ResNet50_stain.ipynb` | _pending_ | _pending_ |

## What this ablation isolates

- Same arch / loss / prompts / split as the baseline; only the input pixels change.
- Reference patch: first benign-colon-tissue image in canonical filename order. Same reference across all three methods.
- Stain normalization applied to **train + test alike** so the model never sees raw colour variability at any stage.
- Per-method $\Delta$ vs the unnormalized baseline isolates two things at once: how much the baseline's accuracy relies on stain-colour shortcut signal, and which method preserves morphological structure best.
