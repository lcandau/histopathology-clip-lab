# exp_06 — Biomedical text encoder ablation (RQ6)

## Question

Within the CLIP paradigm and a frozen-backbone regime, does biomedical text
pretraining (BioBERT, PubMedBERT) help over generic BERT-base under short
class-prompt vocabularies, and how does the prompt format itself interact with
that signal?

## Design

A 3 × 2 ablation matrix. Both axes vary one component at a time relative to
the baseline. A third strategy (`multi_prompt`) is supported by the code but
not run by default — under a frozen text encoder the 5 templates tokenize into
near-identical embeddings, so the augmentation rarely moves the needle on a
5-class closed-set task. Re-enable by adding `"multi_prompt"` back to
`STRATEGIES` if you want the data.

|                  | name_only             | composed (baseline format) |
|------------------|-----------------------|----------------------------|
| **BERT-base**    | bert_name_only        | bert_composed              |
| **BioBERT v1.1** | biobert_name_only     | biobert_composed           |
| **PubMedBERT**   | pubmed_name_only      | pubmed_composed            |

### Text encoders (all loaded via HuggingFace `TFAutoModel`)

- `bert-base-uncased` — generic English (matches the historical baseline but
  re-run through the same HF pipeline so the comparison is apples-to-apples).
- `dmis-lab/biobert-v1.1` — BERT continued on PubMed abstracts + PMC articles.
- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` — pretrained from
  scratch on PubMed abstracts (no general-domain BERT init).

All three output 768-d hidden state, all use BERT-family tokenizers.

### Prompt strategies

- **name_only** — just the LC25000 class name (`"benign lung tissue"`).
- **composed** — `"A histopathology image of {class_name}."` (baseline format).

### Fixed (held identical to the baseline)

- Image side: KerasHub `resnet_50_imagenet` backbone, frozen, 224 × 224.
- Loss: symmetric InfoNCE with a learnable `logit_scale`.
- Optimiser: AdamW, lr 1e-3, wd 1e-4, grad clip 1.0.
- Schedule: max 30 epochs, EarlyStopping(patience=5, restore_best_weights=True).
- Split: same stratified 80/10/10 from `results/splits/lc25000_seed42.json`.
- Embedding dim: 256 with LayerNorm head.
- Initial temperature: τ₀ = 0.07.
- Seed: 42; op-determinism enabled.

The only knobs that move per cell are `(text_encoder_id, prompt_strategy)`.

## Outputs per cell

- `local_drive/<RUN_DIR>/<cell>/weights.weights.h5` — trainable params (projection +
  logit_scale). On Colab this mirrors to the user's Drive cache. The `.weights.h5`
  suffix is required by Keras 3's weights-only save API.
- `results/metrics/exp_06_biomed_text/<cell>_classification.json`
- `results/plots/exp_06_biomed_text/<cell>_umap.png`
- `results/confusion_matrices/exp_06_biomed_text/<cell>_confusion_matrices.png`

## Cross-cell aggregation

- **Comparison table** — 3 × 2 grid of macro-F1 (and per-class deltas vs the
  `bert_composed` anchor).
- **UMAP grid** — 6-panel shared UMAP, one panel per cell, so the tribunal can
  see which encoder + prompt combo separates the five classes cleanest.
- **Prompt-separation diagnostic** — mean pairwise cosine distance between the
  five class-prompt embeddings, by cell. A larger value means the text encoder
  is producing more distinct class prompts (and is a load-bearing precondition
  for InfoNCE to align images correctly).

## Cost estimate

Each cell trains the projection head + a single temperature scalar — the same
small parameter budget as the v7 baseline (~200 K trainable params). On a
Colab L4: ~5–6 min/epoch, ~10 epochs to plateau ⇒ ~30 min/cell.
Total: ~6 × 30 min ≈ 3 h for the full matrix. Cached weights make re-runs
free once trained.

## Pipeline note: why HF for all three encoders

The historical baseline used KerasHub's `bert_base_en_uncased` preset, which
has its own preprocessor + tokenizer. BioBERT and PubMedBERT exist only in the
HuggingFace ecosystem, so the entire RQ6 text side is unified on
`AutoTokenizer` + `TFAutoModel`. The `bert_name_only` / `bert_composed` cells
therefore re-anchor the BERT-base number under the HF pipeline. Any small
drift from the historical 0.6273 macro-F1 baseline is expected and is the
cost of having a clean encoder ablation.

## OOD evaluation

`CLIP_Biomed_OOD_eval.ipynb` (co-located in this folder) runs the 6-cell grid
against NCT-CRC-HE-7K under restricted argmax to the 2 colon classes (NORM ↔
benign colon, TUM ↔ colon adeno). The eval is structurally different from the
`experiments/exp_07_ood_eval/*_OOD_eval.ipynb` notebooks (which compare model
architectures on a single dataset); this one compares 6 within-architecture
text-encoder × prompt cells on a single dataset and is therefore specific to
exp_06. It lives here rather than in `exp_07_ood_eval/` to keep the
exp_06 experiment self-contained.

Headline finding (2026-05-28 run, archived in `runs/2026-05-28_biomed_ood/`):
none of the 6 cells beats the baseline CLIP (0.866 macro-F1) on NCT OOD; all
cells land in 0.81–0.86. Vanilla BERT prefers the `composed` prompt format;
biomedical encoders prefer bare class names — a small prompt-design
observation but not enough to flip any cell to winning.
