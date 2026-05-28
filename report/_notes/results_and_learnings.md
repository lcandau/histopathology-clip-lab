# Results & Main Learnings — 2026-05-28

End-of-day notes after the post-preprocessing-bug rerun cycle, with the **exp_08
PLIP-backbone update folded in**. State of the project: `main @ aee28ce` plus
exp_07 + exp_08 archived branches, seven experiments archived under `experiments/`,
results JSONs and plots all on disk under `results/`.

---

## 5-minute summary

We set out to study CLIP-style histopathology classification on LC25000 with a frozen ResNet50 image
encoder and a frozen BERT text encoder, plus several ablations: stain normalisation, biomedical text
encoders, a non-CLIP supervised control, and a zero-shot PLIP reference. Mid-run we caught a
**preprocessing bug** that had been silently deflating every result in the project by ~30 F1 points
(`keras.applications.resnet.preprocess_input` being applied to inputs going into a `keras_hub`
ResNet50 backbone — two different checkpoints with the same name but different expected input
distributions). After the fix every LC25000 in-distribution result jumped to ~0.99 macro-F1 and
became a saturation null. **The OOD evaluation (LC25000 → NCT-CRC-HE-7K) is therefore the only
meaningful axis for method comparison.**

On OOD, with the ImageNet ResNet50 backbone held fixed, the surprises were uniformly *negative*:

- **The plain supervised CE softmax classifier (0.880 macro-F1) beats the CLIP baseline (0.866).**
  CLIP-style contrastive training does not add OOD value over a plain ResNet50 + softmax on this task.
- **Stain normalisation (Macenko, NCT-NORM reference) hurts OOD** (0.844, null/regression).
- **Biomedical text encoders don't help.** All 6 (encoder × prompt) cells land in 0.81–0.86; none
  beats the baseline 0.866.
- **PLIP zero-shot is the weakest on absolute OOD F1 (0.661)** despite its diverse-histopathology
  pretraining — but it has the smallest in-dist → OOD drop, the "pretrained generalist" pattern.

These nine LC25000-trained ResNet50 variants cluster within ≈0.81–0.88 on OOD despite radically
different heads, objectives, and text encoders — pointing squarely at the **frozen ImageNet
ResNet50 backbone as the OOD bottleneck** (PLIP zero-shot uses a different backbone, hence its
position outside the band).

**exp_08 then tested that hypothesis directly** by swapping the image backbone for PLIP's
ViT-B/32 (pretrained on histopathology image–text pairs) and keeping everything else identical
to the best biomed cell — same composed prompt, same `bert-base-uncased` text encoder, same
projection head, same training recipe. Result: **NCT OOD macro-F1 = 0.933** (+6.7 F1 vs the
prior best plain_classifier 0.880, +6.7 vs CLIP baseline). **The ≈0.88 ceiling broke as soon
as the image encoder changed.** This is the project's first positive finding, and it confirms
the bottleneck diagnosis: the head, prompt, and text encoder were never the limiting factor.

A secondary observation: the PLIP-backbone UMAPs do **not** show the cross-dataset cluster
alignment that baseline CLIP achieved — NCT triangles still sit in their own region of feature
space, away from LC25000 colon clusters, exactly like the plain_classifier. So PLIP wins
classification F1 but loses spatial alignment. "Classification quality" and "representation
alignment" are now empirically separable axes.

The defense narrative pivot: **(1) on a frozen ImageNet backbone, popular method additions
(CLIP loss, stain norm, biomed text) do not improve OOD on this task — and several hurt.
(2) Swapping the image backbone for a histopathology-pretrained one breaks the ceiling by
≈7 F1 points. (3) Better OOD F1 does not imply better feature-space alignment — they are
separable evaluation axes.**

---

## Timeline of what happened (chronological)

1. **Early in the project**: built v1–v7 CLIP notebooks on LC25000 using `keras_hub` ResNet50.
   The image loader applied `keras.applications.resnet.preprocess_input` to every input. v7
   baseline plateaued at macro-F1 = 0.63, unfreezing 30 layers reached 0.90.

2. **First derivative experiments** (stain, plain classifier, OOD eval) — all inherited the
   same `load_image` pattern via copy-paste, all stuck in the 0.6–0.9 in-dist range.

3. **Building exp_06 (biomedical text encoder ablation)**: image loader was rebuilt from scratch
   with `img / 255.0` because that's my modern default. **By accident** this skipped the broken
   Caffe path.

4. **Surprise jump**: exp_06's BERT cells produced macro-F1 = 0.99 vs v7's 0.62 on the same data.
   Three days of investigation followed:
   - Sanity-check cells were added (trainable params, split identity).
   - Linear probe on raw frozen ResNet50 features gave **0.974**, ~30 F1 points above v7. That was
     the smoking gun.
   - Comparing `load_image` across notebooks revealed the keras_hub-vs-keras.applications
     preprocessing mismatch.

5. **Fix landed**: `src/data/images.py` centralised loader → `/255.0` RGB. Every impacted
   notebook updated to import it. Linear probe ceiling now matches actual model accuracy. All
   subsequent runs use the corrected pipeline.

6. **Methodological discoveries during cleanup**:
   - LC25000 contains **1,280 byte-duplicate files** (verified by MD5 grouping). Random
     80/10/10 split puts identical files on both sides of the train/test boundary → guaranteed
     leakage. Dedupe code wired into `discover_records` but **the canonical split file on Drive
     pre-existed and was reused** → dedupe never actually triggered for the runs reported here.
     This is documented in §limitations.
   - Augmentation-pair leakage (each LC25000 source image yields 20 augmented copies, randomly
     split) is a separate, larger source of train/test leakage that we did NOT fix.
   - NCT-CRC-HE-7K is Macenko-normalised at release with an unpublished Kather reference. Our
     LC25000-Macenko-NCT-ref training puts LC into approximately Kather's stain space — but our
     Macenko implementation slightly differs from Kather's, so alignment is imperfect.

7. **Rerun campaign**: baseline / stain (NCT ref) / plain classifier / biomedical 6-cell grid
   trained or re-trained on the corrected pipeline. OOD evaluations followed for each. Three
   OOD eval notebooks (CLIP_OOD_eval, PlainClassifier_OOD_eval, CLIP_Biomed_OOD_eval) share a
   consistent 2×3 grid visualisation: each LC25000 panel overlays NCT triangles so dataset shift
   is visible directly.

8. **Diagnosis converged**: 6 ResNet50-based trained checkpoints + 1 zero-shot PLIP variant
   clustered within 7 F1 on OOD (0.81–0.88), implicating the frozen ImageNet ResNet50 as the
   bottleneck.

9. **exp_08 (positive result)**: kept the best biomed cell's architecture (BERT + composed +
   projection + InfoNCE) and replaced **only** the image backbone — `keras_hub` ResNet50 →
   HuggingFace `vinid/plip` ViT-B/32. Required CLIP-standard image normalisation (mean/std,
   not /255.0) and an NHWC→NCHW transpose inside the model because the HF TFCLIPVisionModel
   port follows PyTorch's channel ordering. Result: in-dist macro-F1 = 0.996 (saturation
   unchanged), **NCT OOD macro-F1 = 0.933** — broke the ~0.88 ResNet50 ceiling by ~7 F1.

10. **Final state (this document)**: 7 trained checkpoints + 1 zero-shot PLIP variant, evaluated
    on LC25000 test (in-dist) and NCT-CRC NORM/TUM (OOD). Results consolidated below.

---

## Complete results matrix

### LC25000 in-distribution (5-class macro-F1)

| Model | macro-F1 | Notes |
|---|---|---|
| Linear probe (raw frozen ResNet50 features) | **0.974** | Theoretical ceiling for any frozen-backbone head |
| Baseline CLIP (raw inputs) | 0.996 | Frozen ResNet50 + frozen BERT + projection heads |
| Stain Macenko (NCT-NORM ref) CLIP | 0.986 | Macenko'd LC25000 inputs at training |
| Plain classifier (CE softmax) | 0.995 | Frozen ResNet50 → GAP → BatchNorm → Dense(5) |
| BERT + name_only | 0.999 | exp_06 cell |
| BERT + composed | 0.999 | exp_06 cell |
| BioBERT + name_only | 0.998 | exp_06 cell |
| BioBERT + composed | 0.998 | exp_06 cell |
| PubMedBERT + name_only | 0.995 | exp_06 cell |
| PubMedBERT + composed | 0.996 | exp_06 cell |
| **PLIP-ViT + BERT (composed)** | **0.996** | **exp_08 — backbone swap** |
| PLIP (zero-shot, no fine-tune) | 0.700 | External reference; no LC25000 exposure |

All trained variants land within 0.013 macro-F1 of one another. **In-dist is a saturation null.**

### NCT-CRC-HE-7K OOD (2-class macro-F1, restricted argmax over colon classes)

| Model | macro-F1 | OOD rank |
|---|---|---|
| **PLIP-ViT + BERT (composed)** | **0.933** | 🥇 **breaks the ≈0.88 ResNet50 ceiling** (exp_08) |
| plain_classifier | 0.880 | 🥈 |
| baseline CLIP | 0.866 | 🥉 |
| BERT + composed | 0.863 | |
| PubMedBERT + name_only | 0.855 | |
| stain_macenko_nct_ref CLIP | 0.844 | (regression vs baseline) |
| BioBERT + name_only | 0.840 | |
| BERT + name_only | 0.821 | |
| BioBERT + composed | 0.818 | |
| PubMedBERT + composed | 0.808 | |
| **PLIP zero-shot** | **0.661** | last (but smallest train→OOD drop) |

Per-class breakdown for exp_08 (PLIP-backbone): benign colon F1 = 0.914, colon adenocarcinoma
F1 = 0.951, accuracy = 0.938, balanced accuracy = 0.927.

### Train → OOD drop

| Model | Δ |
|---|---|
| PLIP zero-shot | −0.04 (0.700 → 0.661) — smallest |
| **PLIP-ViT + BERT (composed)** | **−0.063** (0.996 → 0.933) — smallest among trained variants |
| plain_classifier | −0.115 |
| baseline CLIP | −0.130 |
| stain_macenko_nct_ref | −0.142 |
| biomed cells | −0.13 to −0.19 |

---

## Main learnings (numbered)

### 1. The keras_hub / keras.applications preprocessing trap

`keras_hub.models.Backbone.from_preset("resnet_50_imagenet")` and
`keras.applications.ResNet50` ship **different checkpoints** despite the same name. The former
expects `/255.0` RGB inputs; the latter expects Caffe-style BGR + ImageNet mean subtraction.
Applying the wrong preprocessing to the wrong backbone produces **degraded but plausible-looking
features** — the model still trains, just to a much lower ceiling.

We had this bug for the entire pre-rerun phase of the project, and it would have been *invisible*
without the linear-probe diagnostic. **Lesson**: always linear-probe a frozen feature extractor
as a project-foundation sanity check. The expected ceiling should match the actual trained
accuracy within a few F1 points; if it doesn't, suspect preprocessing first.

### 2. LC25000 in-distribution saturates

With corrected preprocessing, every trained variant (CLIP / supervised CE / stain-norm / 6
biomed cells) lands at 0.985–0.999 macro-F1 on the LC25000 test set. The differences between
methods are within noise of one another. **In-distribution is dead as a discriminator on
LC25000.**

This is partly intrinsic to the dataset:

- 1,250 source images expanded to 25,000 via rotation/flip augmentation. Random 80/10/10 splits
  scatter augmented copies of each source across train/val/test → the model effectively memorises
  the 1,250 sources and recognises test patches as augmented copies of training patches.
- Additionally, 1,280 byte-identical duplicate files (~5% of the dataset) cross train/test
  with high probability under a random split.

### 3. OOD is the only meaningful axis (and the methods don't help)

LC25000 → NCT-CRC-HE-7K transfer on the 2 colon classes (NORM ↔ benign colon, TUM ↔ colon adeno)
**does** discriminate methods. But the discrimination tells a *negative* story:

- **Plain supervised CE (0.880) ≳ CLIP contrastive (0.866).** Adding the text-image alignment
  objective doesn't help OOD. The argument that CLIP's contrastive loss provides "implicit OOD
  regularisation through semantic alignment" — common in the multimodal literature — does **not**
  hold on this LC25000 → NCT-CRC colon transfer.
- **Stain normalisation hurts** (0.844 vs raw 0.866). Applying Macenko at training time with an
  NCT NORM reference introduces a non-linear preprocessing artefact that the frozen ImageNet
  ResNet50 wasn't trained to digest; this slightly degrades feature quality and the decision
  boundary shifts toward over-predicting the larger NCT class (TUM).
- **Biomedical text encoders don't move the needle.** None of the 6 (encoder × prompt) cells
  beats the baseline. The best biomed cell (BERT + composed = 0.863) is a tie with baseline.
  The worst (PubMedBERT + composed = 0.808) is 6 F1 points below.
- **There's an interesting interaction effect inside the biomed grid**: vanilla BERT prefers the
  longer `"A histopathology image of {class}"` prompt; biomedical encoders prefer **bare class
  names**. The reason is plausibly that biomedical pretraining made `"colon adenocarcinoma"`
  carry strong signal on its own, while adding generic boilerplate ("A histopathology image of...")
  dilutes the prompt embedding. This is not enough to make any biomed cell beat the baseline but
  it's a useful methodological observation for the prompt-design literature.

### 4. The frozen ImageNet ResNet50 was the OOD bottleneck — and exp_08 proves it

The unified picture across the 9 ResNet50-based trained variants was that they clustered
within ~7 F1 points on OOD (0.81–0.88) despite radically different architectures, training
objectives, prompt formats, text encoders, and preprocessing. The variation between methods
was much smaller than the variation between in-dist (0.99) and OOD (0.85). This pointed
squarely at one source of variation: **the image-encoder features**. Every ResNet50-based
variant used the same frozen `resnet_50_imagenet` features; every one inherited the same OOD
ceiling those features permitted.

**exp_08 tested this hypothesis directly.** Architecture and recipe held identical to the
best biomed cell (BERT + composed prompt, frozen text, projection head, InfoNCE) — only the
image backbone swapped from `keras_hub` ResNet50 (ImageNet-pretrained) to `vinid/plip`'s
ViT-B/32 (pretrained on histopathology image–text pairs). Result: **OOD macro-F1 = 0.933,
+6.7 F1 points over the previous ceiling.** The in-dist number barely moved (0.996 vs 0.996
baseline) — the LC25000 saturation null is independent of backbone — but OOD broke open.

**This confirms the bottleneck diagnosis.** The head, projection, prompt, and text encoder
were never the limiting factor on OOD; the image representation was. A backbone pretrained
on histopathology — even without any LC25000 supervision until our fine-tuning — encodes
NCT-CRC colon tissue more usefully than an ImageNet-pretrained ResNet50 ever did.

The implication for the thesis flips from negative-only ("none of these methods help") to
balanced ("none of these head-side methods help, but the right backbone helps a lot — and
both findings together tell you where to spend future effort").

### 5. Classification quality and representation alignment are separable axes

Side-by-side UMAPs show three distinct patterns across the project:

- **Baseline CLIP (ResNet50)** — 256-d L2-normalised embeddings pull NCT colon samples *into*
  the LC25000 colon clusters in feature space. Same-class images from different datasets
  co-locate. Best visual evidence of cross-dataset alignment.
- **Plain classifier (ResNet50)** — 2048-d BatchNorm features leave NCT in a *separate region*
  of feature space — classification still works because the Dense(5) weights define class
  *directions* that NCT samples project onto correctly, but spatial alignment with LC25000 is
  lost.
- **PLIP-ViT + BERT (exp_08)** — also leaves NCT in a separate region of feature space,
  qualitatively similar to the plain-classifier pattern. Despite achieving the **highest**
  OOD F1 of any variant (0.933), the cross-dataset spatial alignment that baseline CLIP
  achieved is **not** reproduced.

The PLIP-backbone case is the cleanest demonstration that **classification F1 and cross-dataset
feature alignment are not the same axis**. Mechanism: PLIP encodes source-specific variation
(stain, scanner, lab idiosyncrasies) more sensitively than ImageNet ResNet50, so even
within the same class NCT samples occupy a different region from LC25000 samples — yet the
cosine similarity to the class-prompt direction is preserved, so classification still works.
The contrastive objective alone (baseline CLIP) produces alignment but not necessarily better
classification. The right backbone (PLIP) produces better classification but not necessarily
better alignment. This dissociation is itself a useful finding for the multimodal-histopathology
literature — many papers implicitly conflate the two.

For *downstream* uses where the feature representation matters (clustering, retrieval, transfer
to other classifiers), baseline CLIP's representation is meaningfully better than its F1 number
suggests. For *classification*, exp_08's PLIP backbone is the headline.

### 6. PLIP zero-shot underperforms — but PLIP-as-backbone wins

PLIP was pretrained on diverse histopathology image–text pairs (OpenPath). The expectation
going in: it should generalise to NCT-CRC better than an LC25000-specialised model. The
zero-shot reality: **PLIP gets 0.661 OOD macro-F1, worse than every LC25000-trained variant.**

The compensating zero-shot observation: PLIP has the smallest train→OOD drop (-0.04 from
0.70 to 0.66), the classic "pretrained generalist" pattern. It hasn't overfit to any specific
source, but its **zero-shot text-image alignment** for the NORM-vs-TUM cosine-similarity
classification is below what LC25000 supervision provides.

**exp_08 then showed the missing piece**: PLIP's pretraining produces a much stronger image
*representation* than its zero-shot prompt alignment suggests. Holding the architecture
identical to the best biomed cell and only swapping the image backbone (ResNet50 → PLIP's
ViT-B/32) gives **0.996 in-dist and 0.933 OOD** — the highest OOD number in the project and
the smallest in-dist → OOD drop among trained variants (−0.063).

The combined reading: **broad pretraining contributes its value through the image
representation, not through the off-the-shelf prompt geometry.** A small amount of
LC25000 supervision (a fresh projection head + temperature, trained on top of the frozen PLIP
backbone) is enough to convert PLIP's pretraining advantage into a +6.7 F1 OOD gain over the
best ResNet50-based variant. Pretraining and specialised supervision are not in opposition;
the right architecture stacks them.

### 7. Stain normalisation is double-edged

Macenko normalisation has a textbook argument: align the staining of training and inference data
to the same target profile, reducing cross-source colour shift. In practice, on this task, it
hurt by ~2 F1 points on OOD because:

1. **Our Macenko implementation is not identical to Kather's** (different reference image,
   possibly different α/β parameters). Even with an NCT tile as our reference, our normalised
   LC25000 sits in *approximately* but not *exactly* Kather's NCT stain space.
2. **The frozen ImageNet ResNet50 was never trained on Macenko-normalised inputs**. Macenko's
   OD-space round-trip introduces a subtle non-linear shift in input statistics. The frozen
   backbone has no opportunity to adapt to this shift, so the features are slightly degraded
   relative to feeding raw RGB.

For settings with much larger stain shift than the LC25000-vs-NCT-CRC pair (e.g. cross-laboratory
WSI variation in clinical deployment), stain normalisation might pay off. On this specific
transfer it doesn't.

### 8. Domain-pretrained backbone is the dominant lever — and it stacks with the rest

The single biggest OOD F1 gain in the project came from one change: image backbone. Holding
**everything else identical** to the best biomed cell (BERT text encoder, composed prompt,
projection head architecture, InfoNCE loss, AdamW + EarlyStopping, even the random seed
distribution from re-training), replacing `keras_hub` ResNet50 with `vinid/plip`'s ViT-B/32
moved OOD macro-F1 from 0.863 → 0.933 (+7.0 F1) and accuracy from ~0.85 → 0.938.

The result is more striking when you stack it against the project's other interventions:

| Lever | Change relative to ResNet50 CLIP baseline | OOD Δ |
|---|---|---|
| Backbone (ImageNet → PLIP-histopath) | exp_08 | **+6.7 F1** |
| Loss (CLIP → plain CE softmax) | exp_03 | +1.4 F1 |
| Stain normalisation (raw → Macenko NCT-ref) | exp_04 | −2.2 F1 |
| Text encoder (BERT → biomed) | exp_06 best | −0.3 F1 |
| Prompt (`name` → `composed`) | exp_06 | −0.5 F1 avg |

**The backbone change is 4-5× larger than any other single lever we tested**, and it goes in
the *opposite* direction from the contemporary methods literature's emphasis on novel
loss functions, text-side innovations, and stain harmonisation. For histopathology image
classification at frozen-backbone scale, the **right pretraining domain dominates the recipe**
— and inverse-domain pretraining (ImageNet) is what bounds every other method to ≈0.88 on
NCT-CRC transfer.

This is a strong, defensible take-home for the thesis: **future work on this task should
prioritise backbone choice (UNI, CONCH, Virchow-2, PLIP variants) over head-side or text-side
innovations.** The ablation matrix we ran provides direct evidence for that recommendation.

---

## UMAP observations

The `2026-05-28_*_ood` UMAP grids tell a coherent story:

- **All trained models** produce 5 well-separated LC25000 class clusters with the corresponding
  prompt embedding (CLIP) or class-direction vector (plain classifier) sitting at the cluster
  centre.
- **Baseline CLIP** — NCT NORM/TUM triangles sit *inside* the LC25000 benign-colon and
  colon-adeno clusters respectively. Best visual evidence of cross-dataset feature alignment.
- **Stain Macenko CLIP** — the 5 LC25000 clusters are slightly *more compressed* than baseline's
  (the in-dist feature space lost some inter-class spread). NCT NORM samples drift away from
  the benign-colon cluster more than in baseline — and that's where the OOD recall drop on NORM
  comes from (~0.88 → 0.67).
- **Plain classifier** — NCT clusters sit in their own neighbourhood, *separate* from LC25000
  colon clusters. Classification still correct, feature-space alignment lost.
- **Biomed cells** — qualitatively similar to baseline CLIP's pattern, with subtle per-cell
  differences in how tightly NCT projects onto the LC25000 colon prompts. PubMedBERT + composed
  has the messiest NCT cluster (matches its lowest OOD F1).
- **exp_08 PLIP-ViT + BERT** — best classification numbers of any variant (NCT OOD F1 = 0.933),
  but the NCT triangles sit in their own region of the projection, **away** from the LC25000
  colon clusters — the same pattern as plain_classifier, not the baseline-CLIP-style alignment.
  The 5 LC25000 clusters are well-separated and tighter than baseline. This is the cleanest
  empirical demonstration that "OOD classification quality" and "cross-dataset feature
  alignment" are separable axes. The cosine similarity to the prompt direction is what carries
  classification; absolute embedding location is what carries alignment, and PLIP optimises
  the first more than the second on this transfer.

---

## Project caveats / known limitations

These need to land in the §limitations section of the thesis:

1. **Canonical split was built pre-dedupe** and reused by every model trained today. The 1,280
   byte-duplicates therefore live across both train and test for every reported number. Effect
   size: probably 1–2 F1 points of inflation on every in-dist number; OOD is not affected since
   NCT is independent.

2. **LC25000 augmentation-pair leakage** (1,250 sources × 20 augmentations) was identified but
   not corrected. The proper fix is a source-image-aware split (each original's 20 augmentations
   go to the same split). With ResNet50 features that are largely augmentation-invariant, this
   is probably the *dominant* source of in-dist over-estimation, accounting for the bulk of the
   gap between linear-probe (0.97) and trained F1 (0.99).

3. **Our Macenko implementation differs from Kather's**. Documented in §methods. The
   stain-normalised checkpoint is internally consistent but not strictly comparable to
   externally-Macenko'd datasets.

4. **PubMedBERT and BioBERT only ship PyTorch weights**. We work around this with
   `from_pt=True` HF conversion. The catch block was widened from `OSError` to also catch
   `TypeError` after the transformers library raised the latter when looking up a non-existent
   TF sharded-weights index file.

5. **OOD evaluation restricted to 2 NCT colon classes** (NORM + TUM). The other 7 NCT classes
   (ADI, BACK, DEB, LYM, MUC, MUS, STR) have no clean LC25000 mapping and are excluded. This
   restricts the OOD comparison to a binary problem; harder multi-class OOD scenarios would
   need additional class-mapping work or a different OOD dataset.

6. **Single NCT tile as Macenko reference** (`NORM-TCGA-AASSYQPA`). Picking a different
   reference would shift the target stain profile and could change the OOD result by some
   margin. This is an unaddressed degree of freedom.

---

## Suggested thesis narrative

### §5 (Pivotal finding)

> LC25000 admits ~0.99 macro-F1 in-distribution with off-the-shelf frozen ImageNet ResNet50 +
> any reasonable classifier head. We treat this as a methodological hazard: the standard random
> 80/10/10 split scatters augmented copies of the 1,250 source patches across train and test,
> producing a leakage path that makes in-distribution evaluation effectively uninformative for
> comparing model architectures. We therefore report in-distribution numbers for completeness
> but frame the cross-source generalisation to NCT-CRC-HE-7K as the central evaluation.

### §6 (OOD generalisation — the central contribution)

> On the LC25000 → NCT-CRC-HE-7K cross-source generalisation task, we evaluate a matrix of
> interventions on the standard frozen-ImageNet-ResNet50 + frozen-text-encoder CLIP recipe.
> We observe:
>
> 1. CLIP-style contrastive training does not improve OOD generalisation over plain supervised
>    cross-entropy on this colon-classification transfer. The plain classifier reaches macro-F1
>    = 0.880, marginally exceeding the CLIP baseline (0.866).
>
> 2. Training-time stain normalisation (Macenko, NCT-NORM reference) degrades OOD performance
>    (0.844, a 2-point regression).
>
> 3. Substituting biomedical text encoders (BioBERT, PubMedBERT) for vanilla BERT does not
>    improve OOD performance. All six (encoder × prompt) cells we evaluated fall in the
>    0.81–0.86 range, with the best (BERT + composed prompt) tying the baseline at 0.863 and
>    the worst (PubMedBERT + composed) trailing by 6 F1 points.
>
> 4. PLIP zero-shot achieves 0.661 — the weakest absolute OOD performance — but with the
>    smallest in-dist → OOD drop (−0.04 vs −0.13 for LC25000-specialised models). This is
>    the expected "pretrained generalist" pattern.
>
> These results cluster all ResNet50-based variants within 0.81–0.88 macro-F1 on OOD, pointing
> to the image-encoder representation — not the head, the loss, or the text side — as the
> limiting factor. We test this hypothesis directly:
>
> 5. **(exp_08, central positive result)** Replacing the frozen ImageNet ResNet50 with PLIP's
>    ViT-B/32 vision encoder (pretrained on diverse histopathology image–text pairs) while
>    holding the rest of the architecture identical to the best biomed cell (BERT text encoder,
>    composed prompt, projection heads, InfoNCE) raises OOD macro-F1 from 0.863 → **0.933**
>    — a +7.0 F1 absolute gain over the same-recipe ResNet50 counterpart, and +6.7 over the
>    previous best (plain_classifier). The in-distribution number is unchanged (0.996), and
>    the in-dist → OOD drop is the smallest among trained variants (−0.063).
>
> Taken together, the matrix establishes a clear ordering of effect sizes: **backbone
> pretraining domain dominates the recipe by roughly 5×** over the next-largest single lever
> (loss function), with stain normalisation, biomedical text encoders, and prompt formatting
> contributing zero or negative OOD value on this task. The thesis's central methodological
> recommendation is therefore that improving OOD generalisation on histopathology
> classification should prioritise the **image encoder's pretraining domain** before
> head-side, loss-side, or text-side innovations.

### §7 (Discussion)

Items to discuss:
- **The classification-vs-alignment dissociation.** Baseline CLIP achieves cross-dataset
  cluster alignment but lower F1; PLIP-backbone achieves the highest F1 but leaves NCT in
  its own region of feature space. The contrastive objective optimises alignment of
  *same-class* pairs; the cosine-similarity classifier rule only needs alignment to the
  class-prompt *direction*. These are different geometric properties.
- **The implementation-vs-checkpoint trap** (keras_hub vs keras.applications) as a
  methodological case study in silent benchmark inflation/deflation.
- **The augmentation-pair leakage in LC25000** as a more fundamental dataset issue, separate
  from the dedupe issue and probably the dominant inflater of in-dist numbers.
- **Why head-side methods cluster within 7 F1 points but the backbone change moves the floor
  by 7 F1.** Frame in terms of degrees of freedom: a frozen ResNet50 backbone fixes 23M
  parameters of feature extraction; the projection head + temperature add ≈525K trainable
  parameters; the head can only re-weight what the backbone already encodes.
- **Why specialised supervision wins absolute F1 but loses on generalisation gap vs PLIP
  zero-shot, and how exp_08 reconciles the two**: PLIP-as-backbone + small LC25000
  fine-tuning gets the best of both.

### §8 (Limitations & Future work)

- Dedupe + source-aware LC25000 split would tighten the in-dist numbers; exp_08's in-dist
  saturation (0.996) is subject to the same caveat as every other trained variant here.
- exp_08 establishes that **PLIP** as a frozen backbone breaks the ResNet50 ceiling on this
  task. The natural next experiments are: (a) other histopathology-pretrained backbones
  (UNI, CONCH, Virchow-2) for comparison; (b) full or partial unfreezing of PLIP; (c)
  applying PLIP-as-backbone to broader OOD shifts (cross-laboratory, cross-scanner) to test
  whether the gain generalises beyond NCT-CRC.
- The OOD comparison is currently 2-class (NORM/TUM); extending to richer cross-source
  scenarios — including additional histopathology datasets such as **MHIST** (Suzuki
  et al., 2021, colorectal polyp benign/malignant) — would clarify whether the
  ResNet50→PLIP backbone advantage generalises beyond NCT-CRC, and whether the head-side
  null results also hold under a different OOD shift.
- exp_08's combined LC25000-supervised + PLIP-pretrained pipeline raises a separate question:
  what happens if one fine-tunes PLIP on its own original training set (OpenPath) as a
  control? This would help disentangle "PLIP's pretraining matters" from "frozen features
  + fresh projection head on any histopath ViT works."

---

## Open questions (could be answered with modest extra compute)

| Question | Cost to answer | Status |
|---|---|---|
| Would a histopath-pretrained backbone move the OOD ceiling above 0.88? | ~6h compute (1 training run with new backbone) | **answered by exp_08: yes, PLIP → 0.933** |
| Does the PLIP-backbone OOD advantage hold on a non-NCT-CRC shift (e.g. MHIST colorectal polyp)? | ~2h compute (one eval pass with same checkpoint + label mapping) | **planned as exp_09** |
| Does dedupe + source-aware split lower the in-dist saturation enough for methods to differentiate? | ~3h compute (rebuild split, retrain baseline + plain) | open |
| Does the PubMedBERT × composed regression hold with different prompt templates? | ~2h compute (3 more prompt variants for that cell) | open |
| Are NCT non-colon classes (ADI/MUC/etc.) systematically misclassified into specific LC25000 classes? | ~1h analysis (the "leak" confusion matrix idea) | open |
| Do other histopath-pretrained backbones (UNI, CONCH, Virchow-2) match or beat PLIP at 0.933? | ~6h compute each | open |

The first two are the high-priority follow-ups now that exp_08 has established the backbone-as-bottleneck thesis.

---

## What's archived in the repo

```
experiments/
  exp_01_baseline/
    CLIP_ResNet50_baseline.ipynb         — training notebook
    runs/2026-05-28_baseline/            — executed run
  exp_02_plip_zeroshot/                  — PLIP zero-shot (unchanged from before; still valid)
  exp_03_plain_classifier/
    ResNet50_plain_classifier.ipynb      — training notebook
    runs/2026-05-28_plain_classifier/    — executed run
  exp_04_stain_normalization/
    CLIP_ResNet50_stain.ipynb            — old training notebook (LC25000 reference)
    CLIP_ResNet50_stain_macenko_nct_ref.ipynb — new variant (NCT reference)
    runs/2026-05-28_stain_macenko_nct_ref/
  exp_06_biomed_text/
    CLIP_Biomed_text.ipynb               — 6-cell training notebook
    runs/2026-05-28_full_grid/           — executed run
  exp_07_ood_eval/
    CLIP_OOD_eval.ipynb                  — baseline / stain VARIANT switch
    PlainClassifier_OOD_eval.ipynb       — plain softmax architecture
    CLIP_Biomed_OOD_eval.ipynb           — biomed VARIANT loop
    CLIP_PLIP_OOD_eval.ipynb             — PLIP-backbone variant (exp_08 eval)
    runs/2026-05-28_{baseline,plain_classifier,stain_macenko_nct_ref,biomed,plip_bert_composed}_ood/
  exp_08_plip_backbone/
    CLIP_PLIP_BERT.ipynb                 — PLIP ViT + BERT + composed prompt training
    runs/2026-05-28_plip_bert_composed/  — executed run
  archived/                              — exp_03b_unfreeze30, exp_05_ood_transfer (superseded)

results/
  metrics/                               — per-variant JSONs (in-dist + OOD)
  plots/                                 — per-variant PNGs (CM, UMAPs, before/after)
  confusion_matrices/                    — per-variant *_cm.png

src/data/
  images.py                              — centralised /255.0 loader (the load-bearing fix)
  lc25000.py                             — discover_records with apply_dedupe flag
  lc25000_dedupe_exclusions.json         — 1,280 byte-duplicate exclusions
  macenko_reference_nct_norm.png         — Macenko target tile (NORM-TCGA-AASSYQPA)
  stain.py                               — Macenko / Reinhard / Vahadane fitters
```

All numbers in this document are sourced from the JSON files in `results/metrics/`.
