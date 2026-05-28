# Results & Main Learnings — 2026-05-28

End-of-day notes after the post-preprocessing-bug rerun cycle. State of the project:
`main @ 7b6faa2` (Merge exp_06_biomed_text), six experiments archived under `experiments/`,
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

On OOD, the surprises were uniformly *negative*:

- **The plain supervised CE softmax classifier (0.880 macro-F1) beats the CLIP baseline (0.866).**
  CLIP-style contrastive training does not add OOD value over a plain ResNet50 + softmax on this task.
- **Stain normalisation (Macenko, NCT-NORM reference) hurts OOD** (0.844, null/regression).
- **Biomedical text encoders don't help.** All 6 (encoder × prompt) cells land in 0.81–0.86; none
  beats the baseline 0.866.
- **PLIP zero-shot is the weakest on absolute OOD F1 (0.661)** despite its diverse-histopathology
  pretraining — but it has the smallest in-dist → OOD drop, the "pretrained generalist" pattern.

The honest reading: **the frozen ImageNet ResNet50 backbone is doing essentially all the OOD
generalisation work on this LC25000 → NCT-CRC colon transfer.** Auxiliary methods (CLIP, stain,
biomedical text, prompt engineering) do not contribute additional OOD signal beyond what the
frozen backbone provides — and several actively hurt. This is a strong, contrarian, publishable
finding.

The defense narrative should pivot from *"these methods help OOD"* to *"on this task they
demonstrably don't, and we explain why."*

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

8. **Final state (this document)**: 6 trained checkpoints + 1 zero-shot PLIP variant, evaluated
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
| PLIP (zero-shot, no fine-tune) | 0.700 | External reference; no LC25000 exposure |

All trained variants land within 0.013 macro-F1 of one another. **In-dist is a saturation null.**

### NCT-CRC-HE-7K OOD (2-class macro-F1, restricted argmax over colon classes)

| Model | macro-F1 | OOD rank |
|---|---|---|
| **plain_classifier** | **0.880** | 🥇 |
| **baseline CLIP** | 0.866 | 🥈 |
| BERT + composed | 0.863 | 🥉 |
| PubMedBERT + name_only | 0.855 | |
| stain_macenko_nct_ref CLIP | 0.844 | (regression vs baseline) |
| BioBERT + name_only | 0.840 | |
| BERT + name_only | 0.821 | |
| BioBERT + composed | 0.818 | |
| PubMedBERT + composed | 0.808 | |
| **PLIP zero-shot** | **0.661** | last (but smallest train→OOD drop) |

### Train → OOD drop

| Model | Δ |
|---|---|
| PLIP zero-shot | −0.04 (0.700 → 0.661) — smallest |
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

### 4. The frozen ImageNet ResNet50 explains nearly all the generalisation

The unified picture across all 10 trained-or-zero-shot variants is that they cluster within ~7
F1 points on OOD (0.81–0.88) despite radically different architectures, training objectives,
prompt formats, text encoders, and preprocessing. The variation between methods is much smaller
than the variation between in-dist (0.99) and OOD (0.85). This points squarely at one source
of variation: **the image-encoder features**. Every variant uses the same frozen
`resnet_50_imagenet` features; every variant inherits the same OOD ceiling that those features
permit.

**Implication for the thesis**: the bottleneck for OOD transfer on this task is the **image
representation**, not the classification head or the auxiliary training objective. To improve
OOD beyond ~0.88 macro-F1 on this task you would need a different image backbone — likely one
pretrained on histopathology (e.g. UNI, CONCH, Virchow-2) — not a different head.

### 5. CLIP's feature geometry is different from plain softmax's, even when their accuracy isn't

Side-by-side UMAPs show:

- **CLIP**'s 256-d L2-normalised embeddings pull NCT colon samples *into* the LC25000 colon
  clusters in feature space. Same-class images from different datasets co-locate.
- **Plain classifier**'s 2048-d BatchNorm features leave NCT in a *separate region* of feature
  space — classification still works because the Dense(5) weights define class *directions*
  that NCT samples project onto correctly, but spatial alignment with LC25000 is lost.

This is the contrastive objective doing what it advertises (cluster same-class images together
regardless of source) — it's just that this geometric improvement doesn't translate into higher
classification F1 on this 2-class colon OOD task. For *downstream* uses where the feature
representation matters (clustering, retrieval, transfer to other classifiers), CLIP's
representation is meaningfully better even though it tied/lost on raw F1.

### 6. PLIP underperforms despite the broader pretraining

PLIP was pretrained on diverse histopathology image-text pairs (OpenPath). The expectation
going in: it should generalise to NCT-CRC better than an LC25000-specialised model. The
reality: **PLIP zero-shot gets 0.661 OOD macro-F1, worse than every LC25000-trained variant.**

The compensating observation: PLIP has the smallest train→OOD drop (-0.04 from 0.70 to 0.66),
the classic "pretrained generalist" pattern. It hasn't overfit to any specific source, but its
absolute representation quality for the colon NORM-vs-TUM task is below LC25000-specialised
training.

This isn't a knock on PLIP — it's evidence that **broad pretraining and specialised supervision
optimise different things**, and which one wins on a given downstream depends on the downstream.
On this specific NORM-vs-TUM colon task, specialised supervision (whether CLIP or plain) on a
single source (LC25000) wins on raw F1, but pays an OOD-shift cost that pretraining sidesteps.

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

> On the LC25000 → NCT-CRC-HE-7K cross-source generalisation task, we observe a pattern that
> contradicts a common implicit assumption in the multimodal histopathology literature.
> Specifically:
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
> 4. PLIP, pretrained on diverse histopathology image-text pairs, achieves 0.661 zero-shot — the
>    weakest absolute OOD performance — but with the smallest in-dist → OOD drop (−0.04 vs −0.13
>    for LC25000-specialised models). This is the expected "pretrained generalist" pattern.
>
> Taken together, these results indicate that on this task the bottleneck for OOD transfer is
> not the classification head, the training objective, or the prompt format, but the
> **image-encoder representation itself**. Improving OOD generalisation beyond the observed
> ceiling of ≈0.88 would require a different image backbone — specifically, one pretrained on
> histopathology rather than ImageNet.

### §7 (Discussion)

Items to discuss:
- Why CLIP's feature-space alignment (visible in UMAPs) does not translate to F1.
- The implementation-vs-checkpoint trap (keras_hub vs keras.applications) as a methodological
  case study.
- The augmentation-pair leakage in LC25000 as a more fundamental dataset issue.
- Why specialised supervision wins absolute F1 but loses on generalisation gap vs PLIP.

### §8 (Limitations & Future work)

- Dedupe + source-aware split would tighten the in-dist numbers.
- Histopathology-pretrained backbones (UNI, CONCH, Virchow-2) are the obvious next move if the
  goal is to actually beat ~0.88 OOD F1.
- The OOD comparison is currently 2-class (NORM/TUM); extending to richer cross-source
  scenarios would clarify whether the negative findings here generalise.

---

## Open questions (could be answered with modest extra compute)

| Question | Cost to answer |
|---|---|
| Does dedupe + source-aware split lower the in-dist saturation enough for methods to differentiate? | ~3h compute (rebuild split, retrain baseline + plain) |
| Would a histopath-pretrained backbone (UNI, CONCH) move the OOD ceiling above 0.88? | ~6h compute (1 training run with new backbone) |
| Does the PubMedBERT × composed regression hold with different prompt templates? | ~2h compute (3 more prompt variants for that cell) |
| Are NCT non-colon classes (ADI/MUC/etc.) systematically misclassified into specific LC25000 classes? | ~1h analysis (the "leak" confusion matrix idea) |

All four are low-risk, high-information; any one of them strengthens the thesis if pursued.

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
    runs/2026-05-28_{baseline,plain_classifier,stain_macenko_nct_ref,biomed}_ood/
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
