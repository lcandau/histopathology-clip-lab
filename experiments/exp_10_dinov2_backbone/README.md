# exp_10 — DINOv2 vision encoder + BERT text encoder

CLIP-style training on LC25000 with the image backbone swapped from `keras_hub`'s ImageNet-pretrained ResNet50 (used by the exp_01 baseline and all derivatives) to **DINOv2's ViT-B/14 vision encoder** (`facebook/dinov2-base` on HuggingFace, self-supervised on the general-domain LVD-142M image dataset).

The text side is identical to the best-performing biomed cell from exp_06 and to the exp_08 PLIP-backbone experiment:
- HF `bert-base-uncased`, frozen
- Composed prompt: `"A histopathology image of {class_name}."`

Architecture is otherwise the same as exp_08's `CLIPModel_PLIP`:
- Frozen image and text backbones
- `Dense(256, no bias) → LayerNorm` projection head on each side
- Learnable `logit_scale` temperature scalar
- Symmetric InfoNCE loss
- AdamW, lr=1e-3, weight_decay=1e-4, 30 epochs, EarlyStopping(patience=5, restore_best_weights=True)

## What this experiment answers

exp_08 swapped ResNet50 for PLIP and showed an OOD improvement on the NCT/Chaoyang/LungHist700 transfer set. PLIP differs from the ResNet50 baseline in **two** ways simultaneously:

1. **Histopathology pretraining** (PLIP was trained on histo image-text pairs).
2. **ViT + self-supervised** architecture (PLIP uses a CLIP ViT-B/32 vision tower instead of a CNN).

exp_10 is the control that **disambiguates the two**. DINOv2 is a same-family self-supervised ViT pretrained on **general** (non-medical) images. It is the natural reference point for "ViT/SSL with no histo bias":

| Variant | Architecture | Pretraining domain |
|---|---|---|
| baseline (exp_01) | ResNet50 | ImageNet (general) |
| plip_bert_composed (exp_08) | ViT-B/32 | Histopathology |
| **dinov2_bert_composed (exp_10)** | **ViT-B/14** | **General (LVD-142M)** |

If exp_10 OOD F1 sits close to exp_08, most of PLIP's advantage is architectural / SSL. If exp_10 lands closer to the exp_01 ResNet50 baseline, the bottleneck really is **histopathology pretraining**, and architecture is secondary.

## Two-step workflow: precompute features, then train

`transformers==4.46.0` (the version pinned across the rest of the project, see `requirements.txt`) **does not provide a TensorFlow port for DINOv2**:

```
ValueError: Unrecognized configuration class
<class 'transformers.models.dinov2.configuration_dinov2.Dinov2Config'>
for this kind of AutoModel: TFAutoModel.
```

Since the DINOv2 backbone is **frozen** in this design, we only need to forward each image through it once. We split exp_10 into two notebooks:

1. **`DINOv2_feature_precompute.ipynb`** — PyTorch-only notebook that runs `facebook/dinov2-base` over the four datasets we need (LC25000, NCT-CRC NORM/TUM, Chaoyang normal/adenocarcinoma, LungHist700 @ 20× nor/aca/scc) and caches the 768-d CLS feature per image to Google Drive as HDF5. Run this once.
2. **`CLIP_DINOv2_BERT.ipynb`** — the TF/Keras training notebook. Reads features from `lc25000.h5`, trains the projection heads + temperature on InfoNCE, saves `weights.weights.h5`. No DINOv2 model is ever constructed in this notebook.
3. The three OOD eval notebooks in `../exp_07_ood_eval/` (NCT / Chaoyang / LungHist700) read both the dataset-specific cache **and** `lc25000.h5` for the UMAP reference embeddings.

## Cache layout

Cache root on Drive:
```
/content/drive/MyDrive/clip_histopathology/cache/dinov2/
├── lc25000.h5             # ~25k images × 768d  ≈ 75 MB (gzip level 4)
├── nct_crc.h5             # NCT NORM + TUM (~2k images)
├── chaoyang.h5            # Chaoyang normal + adenocarcinoma (~4k images, train+test)
└── lunghist700_20x.h5     # LungHist700 @ 20× nor/aca/scc (~360 images)
```

HDF5 schema (per file):
| Dataset/Attribute | Type | Description |
|---|---|---|
| `features` | `(N, 768)` float32, gzip-4 | DINOv2 ViT-B/14 CLS token |
| `paths` | `(N,)` S512 | per-image key (see below) |
| attr `model_id` | str | `"facebook/dinov2-base"` |
| attr `img_size` | int | `224` |
| attr `n` | int | N |
| attr `key_kind` | str | `"relative_path"` |
| attr `dataset_tag` | str | one of `lc25000` / `nct_crc` / `chaoyang` / `lunghist700_20x` |

## Cache key scheme

The `paths` dataset stores **per-image relative POSIX paths under each dataset's natural root**, not absolute paths. This keeps the cache portable across Drive locations.

| Dataset | Root | Example key |
|---|---|---|
| LC25000 | `<kagglehub-root>/lung_colon_image_set/` | `lung_image_sets/lung_n/lungn1.jpeg` |
| NCT-CRC | `<extract-root>/CRC-VAL-HE-7K/` | `NORM/CRC-VAL-...tif` |
| Chaoyang | `<drive>/.../chaoyang-data/` | `train/535940-IMG009x022-2.JPG` |
| LungHist700 | `<drive>/.../LungHist700/` (flat) | `aca_wd_20x_001.jpg` (basename) |

LungHist700 uses the bare basename because the release is effectively flat (no class subfolders) and basenames are already unique.

Downstream notebooks resolve each absolute path at training/eval time via the same convention and look up `_index[key]` in the HDF5 file. The eval notebook's `_dinov2_lookup(path)` falls back to LungHist-style basename matching if the relative-path lookup misses.

## Image preprocessing

DINOv2's standard preprocessing (resize 224×224 + ImageNet mean/std) happens **inside the precompute notebook** via HF's `AutoImageProcessor` — the official recipe. The TF notebooks never decode pixels for the DINOv2 variant.

## Cache invalidation

To force recomputation:
```python
# In Colab
!rm /content/drive/MyDrive/clip_histopathology/cache/dinov2/*.h5
```
Then re-run `DINOv2_feature_precompute.ipynb`. Each per-dataset cell short-circuits when its `.h5` already exists, so partial recomputation is just deleting the file(s) you want to regenerate.

## What the training notebook saves

Since the vision encoder lives in the precompute cache, the trained weights file only contains the projection heads + `logit_scale` (~400K parameters, the same scale as exp_08's `plip_bert_composed` checkpoint):
- `<RUN_DIR>/weights.weights.h5`
- `<RUN_DIR>/history.json`

## See also

- `../exp_08_plip_backbone/README.md` — same architecture except the vision backbone (PLIP ViT-B/32 with pre-existing TF port). Read that for the rest of the design rationale.
- `../exp_07_ood_eval/{NCT,Chaoyang,LungHist700}_OOD_eval.ipynb` — 4-variant OOD eval; `VARIANT = "dinov2_bert_composed"` is the relevant branch.
