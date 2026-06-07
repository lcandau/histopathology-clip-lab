# exp_07 — OOD evaluation suite (LC25000-trained → 3 external datasets, 2 organs)

Out-of-distribution evaluation of LC25000-trained checkpoints against three external histopathology
datasets across colon and lung. Restricted-argmax classification against the LC25000-mappable
classes per dataset; per-class metrics + 2×3 UMAP grids + confusion matrices saved under `results/`.

## Naming convention (post-2026-05-29)

This folder uses two conventions and they overlap. The **active** notebooks follow the dataset-first
convention: one notebook per OOD dataset, each with a `VARIANT` switch over the model architectures
to evaluate. The **legacy** notebooks (pre-exp_09) followed a model-first convention and live under
`legacy/`; they are kept for archived-run reproducibility but new evaluation work happens in the
top-level notebooks.

## Active notebooks (dataset-first, multi-variant)

| Notebook | OOD dataset | Classes scored | VARIANT switch |
|---|---|---|---|
| `NCT_OOD_eval.ipynb`         | NCT-CRC-HE-7K (Kather 2019)        | 2 colon (NORM, TUM)              | baseline / plain_classifier / plip_bert_composed / dinov2_bert_composed |
| `Chaoyang_OOD_eval.ipynb`    | Chaoyang (Zhu et al. 2021)         | 2 colon (normal, adenocarcinoma) | same 4 variants |
| `LungHist700_OOD_eval.ipynb` | LungHist700 (Tabatabaei et al. 2024) | 3 lung (normal, AdC, SqC)      | same 4 variants |

Each notebook resolves checkpoint paths per VARIANT against the standard Drive runs root and
produces `{variant}_{dataset}_ood_*.{json,png}` under `results/`.

## Legacy notebooks (model-first, single-variant, NCT only)

Under `legacy/`. The archived runs each notebook produced live in `runs/2026-05-28_*/`; those
should still be reproducible from the legacy files.

| Notebook | Scope |
|---|---|
| `legacy/CLIP_OOD_eval.ipynb`         | NCT eval for baseline + stain_macenko_nct_ref variants |
| `legacy/PlainClassifier_OOD_eval.ipynb` | NCT eval for the plain CE softmax classifier |
| `legacy/CLIP_PLIP_OOD_eval.ipynb`    | NCT eval for the PLIP-backbone variant (exp_08) |

The active `NCT_OOD_eval.ipynb` covers the same NCT scoring with the cleaner multi-VARIANT
interface; the legacy notebooks survive primarily so the original archived `runs/2026-05-28_*/`
notebooks remain consistent with their source.

The exp_06 biomedical text-encoder OOD eval (6 cells × NCT) lives with its training experiment
at `experiments/exp_06_biomed_text/CLIP_Biomed_OOD_eval.ipynb` rather than here — it's a
within-architecture ablation specific to exp_06, structurally different from the
cross-architecture eval pattern this folder now houses.

## Preprocessing per VARIANT (active notebooks)

| Variant | Image normalisation | Image encoder |
|---|---|---|
| baseline / plain_classifier              | `/255.0` RGB                                       | `keras_hub` ResNet50 (ImageNet pretrained, frozen) |
| plip_bert_composed                       | CLIP mean/std `[0.481, 0.458, 0.408] / [0.269, 0.261, 0.276]` | `vinid/plip` ViT-B/32 (frozen) |
| dinov2_bert_composed                     | ImageNet mean/std `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]` | `facebook/dinov2-base` ViT-B/14 (frozen) |

All variants use the same trained projection head (Dense(256, no_bias) → LayerNorm) and the same
frozen `bert-base-uncased` text encoder with the composed prompt format
`"A histopathology image of {class_name}."`. Plain classifier is the only exception — it has no
text encoder and outputs 5-way softmax directly.

## Outputs

Per variant × dataset:
- `results/metrics/exp_07_ood_eval/{variant}_{dataset}_ood_classification.json`
- `results/plots/exp_07_ood_eval/{variant}_{dataset}_ood_cm.png`
- `results/plots/exp_07_ood_eval/{variant}_{dataset}_ood_umap_grid.png`

Executed notebook copies land in `runs/<YYYY-MM-DD>_<variant>_<dataset>_ood/`.
