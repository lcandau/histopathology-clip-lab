# Defense study guide

Concepts to review before the defense on 2026-07-16. Ranked by tribunal-question likelihood.

## Deep-dive topics

### 1. UMAP
Non-linear dimensionality reduction. McInnes, Healy & Melville, 2018.

- Builds a weighted `n_neighbors`-graph in high-D (edges weighted by decaying probability); optimizes a low-D layout so its edge weights match, minimizing cross-entropy via SGD.
- Key hyperparameters: `n_neighbors` (local vs global structure), `min_dist` (cluster tightness), `metric` (we use **cosine** because our embeddings are L2-normalized).
- **What it IS good at**: preserving local neighbourhoods.
- **What it's NOT**: distances between clusters are not interpretable as similarities.
- Defense phrasing: "El UMAP no prueba el mecanismo; muestra el patrón que la métrica resume."
- Trap: "Why UMAP and not t-SNE?" → t-SNE over-clusters and loses global structure. UMAP is faster and preserves both. Both would give qualitatively similar results here.

### 2. Macro-F1 vs Accuracy
- F1 = harmonic mean of precision and recall.
  - Precision = TP/(TP+FP); Recall = TP/(TP+FN).
- **Macro** averages per-class F1 equally; **Weighted** by support; **Micro** ≡ accuracy in single-label multi-class.
- We use macro-F1 because our external sets are imbalanced. If the baseline predicts SCC for everything on LungHist700 (128/359 = 36% accuracy), macro-F1 = 0.305 exposes the collapse; accuracy doesn't.
- Trap: "If you cared about cancer detection, wouldn't recall on malignant class be more relevant?" → Yes, in clinical deployment. Macro-F1 is a research summary metric.

### 3. Models

**CLIP** — Radford et al., ICML 2021. Image + text encoders trained contrastively on ~400M web pairs. Enables zero-shot classification via cosine similarity to class-prompt embeddings.

**ResNet50** — He et al., CVPR 2016. 50-layer CNN with residual connections. Pretrained on ImageNet-1k. Our baseline image encoder (frozen).

**ViT** — Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021. Applies Transformer encoder to sequences of image patches (16×16 or 32×32), flattens each, projects to embedding, adds position encodings. Global receptive field from layer 1; data-hungry compared to CNNs. Both PLIP (B/32) and DINOv2 (B/14) are ViTs.

**BERT** — Devlin et al., NAACL 2019. Transformer encoder pretrained by masked-language-modelling on Wikipedia + BooksCorpus. `bert-base-uncased`: 12 layers, 768-d hidden. Our text encoder (frozen).

**BioBERT** — Lee et al., Bioinformatics 2020. BERT further pre-trained on PubMed abstracts + PMC full-text. **Same vocabulary as BERT** — so it fragments biomedical terms the same way.

**PubMedBERT** — Gu et al., ACM Trans. Comp. Healthcare 2021. Trained from scratch on PubMed. Its vocabulary was built from biomedical text, so `adenocarcinoma`, `squamous`, `histopathology` are single tokens.

**DINOv2** — Oquab et al., TMLR 2024. Self-supervised ViT trained on LVD-142M with self-distillation (momentum-EMA teacher, multi-crop, no labels). Our "ViT + SSL general" control in RQ5.

**PLIP** — Huang et al., Nature Medicine 2023. CLIP architecture (ViT-B/32 + BERT) trained on ~208K image-text pairs scraped from medical Twitter (OpenPath). The domain-pretrained arm of RQ5.

Trap on PLIP: "How does PLIP differ from CLIP architecturally?" → It doesn't. Same arch. Different pretraining data. PLIP is ViT-B/32 while DINOv2 is ViT-B/14 — different patch sizes are a confound (see slide 24 limitations).

## Other high-priority concepts

- **InfoNCE loss** — Oord et al. 2018. `L = -log( exp(sim(i,t_i)/τ) / Σ_j exp(sim(i,t_j)/τ) )`. Temperature `τ`: higher = softer, lower = harder. Our `τ` is learned.
- **Restricted argmax / zero-shot** — evaluation protocol: compute cosine of image embed to each class prompt embed, argmax over an **organ-specific subset** of classes.
- **Freezing vs fine-tuning** — why frozen? Isolates the projection-head effect, controls compute, avoids catastrophic forgetting. Trap: "Would fine-tuning change your conclusion?" — Honest answer: possibly; that's in the limitations.
- **Distribution shift taxonomy** — covariate shift (P(x) changes), label shift (P(y) changes), concept shift (P(y|x) changes). Ours is primarily covariate + institutional.
- **Statistical significance / bootstrap** — we bootstrapped CIs on LungHist700 predictions (N=359, single seed). Width of the CI matters for the +0.335 claim.
- **Confusion matrix reading** — row-normalized: diagonal = recall; off-diagonal = confusion patterns. Practice reading yours out loud.

## Medium priority

- **Attention / ViT mechanics** — `softmax(QKᵀ/√d)V`. Global receptive field, data-hungry.
- **Contrastive learning family** — SimCLR, MoCo, BYOL, DINO. Where DINOv2 sits.
- **Stain normalization mechanics** — Macenko: decomposes RGB into stain vectors in optical density (OD) space. Reinhard: colour matching in LAB. Vahadane: sparse non-negative matrix factorization in OD.
- **Adam vs SGD** — Adam adds per-parameter adaptive learning rates + momentum.

## Lower priority

- **UMAP vs t-SNE vs PCA** — trade-offs.
- **BERT tokenization** — WordPiece; `[CLS]`, `[SEP]` special tokens.
- **LC25000 augmentation issue** — near-duplicate pairs from rotation/reflection inflate in-distribution results (fixed in the split via dedup, but 0.996 still reflects it).
- **Dataset citations** — LC25000 (Borkowski 2019), NCT-CRC (Kather 2018-19), Chaoyang (Zhu 2021), LungHist700 (Reyes 2024).

## Traps I've flagged in speaker notes

- Slide 13: jitter of ~10⁻³ across re-runs; deltas smaller than that = ties in noise. Also why "BERT + composed" == baseline (canonical value used).
- Slide 20: PLIP vs DINOv2 mixes objective + patch size + domain — not a factorial isolation.
- Slide 21: UMAP is consistent with metric, does not prove mechanism.
- Slide 24: encoders frozen, LC25000 in-dist inflated, one seed per condition, N=359 for the headline delta.
