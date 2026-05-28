# exp_08 — PLIP vision encoder + BERT text encoder

CLIP-style training on LC25000 with the image backbone swapped from `keras_hub`'s ImageNet-pretrained ResNet50 (used by exp_01 baseline and all derivatives) to **PLIP's ViT-B/32 vision encoder** (`vinid/plip` on HuggingFace, pretrained on diverse histopathology image-text pairs).

The text side is identical to the best-performing biomed cell from exp_06:
- HF `bert-base-uncased`, frozen
- Composed prompt: `"A histopathology image of {class_name}."`

Architecture is otherwise the same as exp_06's `CLIPModel_HF`:
- Frozen image and text backbones
- `Dense(256, no bias) → LayerNorm` projection head on each side
- Learnable `logit_scale` temperature scalar
- Symmetric InfoNCE loss
- AdamW, lr=1e-3, weight_decay=1e-4, 30 epochs, EarlyStopping(patience=5)

## What this experiment answers

The exp_07 OOD results showed all LC25000-trained variants clustering in macro-F1 0.81–0.88 on NCT-CRC, with the plain supervised classifier (0.880) marginally beating CLIP baseline (0.866). We hypothesised the **frozen ImageNet ResNet50 image features are the OOD bottleneck** — auxiliary head/text-side methods cannot move the needle because they all sit on top of the same image representation.

exp_08 tests this directly by swapping the image backbone for one pretrained on histopathology. If the bottleneck hypothesis is right, exp_08 should break the ≈0.88 OOD ceiling. If exp_08 OOD F1 also lands in the 0.81–0.88 range, the bottleneck is something else (label noise, NCT class difficulty, etc.).

## Image preprocessing

PLIP's vision encoder expects CLIP's standard normalisation:
- Resize to 224×224 (bilinear)
- Convert to RGB float32 in [0, 1]
- Subtract mean `[0.48145466, 0.4578275, 0.40821073]`
- Divide by std `[0.26862954, 0.26130258, 0.27577711]`

This is **NOT** the `/255.0` preprocessing used by the rest of the project (which targets the keras_hub ResNet50). exp_08's `load_image_plip` function in the training notebook handles this; no shared loader exists yet because exp_08 is the first experiment to need it.

## Files

- `CLIP_PLIP_BERT.ipynb` — training notebook (this experiment)
- `../exp_07_ood_eval/CLIP_PLIP_OOD_eval.ipynb` — OOD evaluation against NCT-CRC-HE-7K

Both bootstrap from the `exp_08_plip_backbone` branch (with main fallback after eventual merge).
