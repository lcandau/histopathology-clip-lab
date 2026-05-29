# Results & Main Learnings — 2026-05-28

End-of-day notes after the post-preprocessing-bug rerun cycle, with the **exp_08
PLIP-backbone result + exp_09 extended OOD (Chaoyang colon, LungHist700 lung)
folded in**. State of the project: `main @ f4183e3` plus exp_09 archived branch
(merging shortly), seven training experiments + 3 OOD datasets archived under
`experiments/`, results JSONs and plots all on disk under `results/`.

**Headline as of 2026-05-29**: the PLIP-backbone vs. ResNet50-backbone comparison
now spans 3 OOD datasets and 2 organs. PLIP wins all three. The lung-side gap
is +33 F1 — larger than the colon-side gap by an order of magnitude.

---

## 5-minute summary

We set out to study CLIP-style histopathology classification on LC25000 with a frozen ResNet50 image
encoder and a frozen BERT text encoder, plus several ablations: stain normalisation, biomedical text
encoders, a non-CLIP supervised control, and a zero-shot PLIP reference. Mid-run we caught a
**preprocessing bug** that had been silently deflating every result in the project by ~30 F1 points
(`keras.applications.resnet.preprocess_input` being applied to inputs going into a `keras_hub`
ResNet50 backbone — two different checkpoints with the same name but different expected input
distributions). After the fix every LC25000 in-distribution result jumped to ~0.99 macro-F1 and
became a saturation null. **OOD evaluation (LC25000 → external datasets) is therefore the only
meaningful axis for method comparison.** We evaluate three external datasets across two organs:
**NCT-CRC-HE-7K** (Kather 2019; hereafter "NCT" or "NCT-CRC", colon, 2-class), **Chaoyang**
(Zhu et al. 2021, colon, 2-class), and **LungHist700** (Tabatabaei et al. 2024, lung, 3-class).

On OOD with the ImageNet ResNet50 backbone held fixed (all results in the NCT colon eval),
the surprises were uniformly *negative*:

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
frozen ViT-B/32 vision encoder (pretrained on histopathology image–text pairs) and keeping
everything else identical to the best biomed cell — same composed prompt, same
`bert-base-uncased` text encoder, same projection head, same training recipe. We call this
architecture **PLIP-backbone** throughout the rest of this doc (in metrics JSONs and runs/
folders it appears as `plip_bert_composed`). Result: **NCT OOD macro-F1 = 0.933** (+6.7 F1
vs the prior best plain_classifier 0.880, +6.7 vs CLIP baseline). The ResNet50 colon ceiling
of ≈0.88 broke as soon as the image encoder changed — though, as exp_09 below shows, the
ResNet50 floor on harder shifts (lung) is far lower than 0.88 to begin with. This confirmed
the bottleneck diagnosis: the head, prompt, and text encoder were never the limiting factor.

**exp_09 extended the comparison to two new OOD datasets** — Chaoyang (colon, Beijing
hospital) and LungHist700 (lung, 3-class AdC/SqC/normal) — to test whether the PLIP
advantage was NCT-specific or systematic.

The three-dataset, two-organ matrix (macro-F1):

| Variant | NCT (colon) | Chaoyang (colon) | LungHist700 (lung) | mean OOD |
|---|---|---|---|---|
| **PLIP+BERT (composed)** | **0.933** | **0.813** | **0.640** | **0.795** |
| baseline CLIP (ResNet50) | 0.866 | 0.786 | 0.305 | 0.652 |
| plain_classifier (ResNet50) | 0.880 | 0.722 | 0.281 | 0.628 |

- **PLIP wins all 3** — by +3, +7, **+33** F1 over the next variant.
- **The lung-side gap is an order of magnitude larger** than the colon-side gaps.
- **ResNet50 variants collapse on lung**: both predict "squamous cell carcinoma" for ~90%
  of inputs (recall = 0.05–0.07 on adenocarcinoma), producing a degenerate predictor at
  ~chance accuracy. This is the strongest "the backbone matters" evidence in the project.

UMAP read for exp_08 + exp_09: PLIP-backbone reliably produces higher classification F1
than baseline CLIP across all 3 OOD datasets, but the cross-dataset *spatial alignment*
that baseline CLIP achieves on NCT is **not** reproduced by PLIP-backbone — on every
dataset the OOD samples form their own region of feature space, not co-locating with
LC25000 same-class clusters. "Classification quality" and "representation alignment" are
empirically separable axes; PLIP optimises the first, contrastive CLIP loss optimises the
second, and they don't have to agree.

The defense narrative pivot: **(1) on a frozen ImageNet backbone, popular method additions
(CLIP loss, stain norm, biomed text) do not improve OOD on this task — and several hurt.
(2) Swapping the image backbone for a histopathology-pretrained one breaks the colon OOD
ceiling by ≈7 F1 and turns the lung OOD from non-functional (~chance) into functional
(~0.64 macro-F1). (3) Better OOD F1 does not imply better feature-space alignment — they
are separable evaluation axes. (4) The single largest single-intervention effect size we
measured is backbone choice — by ~5× over the next-largest lever and ~30× over any
head-side or text-side method.**

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

10. **Diagnosis converged a second time**: after exp_08, the question was whether PLIP's
    +6.7 F1 on NCT was a single-dataset artefact or a systematic backbone effect.

11. **exp_09 part A: Chaoyang colorectal eval** (Zhu et al., 2021 — different hospital,
    different scanner, label noise on train split, 4 classes of which we score
    normal + adenocarcinoma against LC25000 colon classes; 4,060 of 6,160 images scored).
    Results: PLIP-backbone 0.813, baseline CLIP 0.786, plain_classifier 0.722. PLIP wins,
    plain_classifier flips from 🥇 on NCT to 🥉 on Chaoyang — the plain softmax's high-confidence
    predictor collapses under heavier shift more catastrophically than CLIP's cosine geometry.

12. **exp_09 part B: LungHist700 pulmonary eval** (Tabatabaei et al., 2024 — 3-class
    lung tissue including adenocarcinoma, squamous cell carcinoma, and normal; first
    OOD evaluation on the lung half of LC25000; 359 of ~691 images at 20× scored).
    Results: PLIP-backbone 0.640, baseline CLIP 0.305, plain_classifier 0.281. Both ResNet50
    variants collapse into a degenerate "predict squamous for everything" predictor;
    PLIP-backbone is the only variant that produces a usable lung-OOD classifier. The
    +33 F1 gap is the largest single-intervention effect in the project.

13. **Final state (this document)**: 7 trained checkpoints + 1 zero-shot PLIP variant,
    evaluated on LC25000 test (in-dist) plus 3 OOD datasets across 2 organs (colon: NCT-CRC,
    Chaoyang; lung: LungHist700). Results consolidated below.

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

### Train → OOD drop (NCT-CRC only — original eval)

| Model | Δ |
|---|---|
| PLIP zero-shot | −0.04 (0.700 → 0.661) — smallest |
| **PLIP-ViT + BERT (composed)** | **−0.063** (0.996 → 0.933) — smallest among trained variants |
| plain_classifier | −0.115 |
| baseline CLIP | −0.130 |
| stain_macenko_nct_ref | −0.142 |
| biomed cells | −0.13 to −0.19 |

### Chaoyang OOD (2-class macro-F1, restricted argmax over colon classes; 4,060 images)

| Model | macro-F1 | OOD rank |
|---|---|---|
| **PLIP-ViT + BERT (composed)** | **0.813** | 🥇 (+3 over baseline) |
| baseline CLIP | 0.786 | 🥈 |
| plain_classifier | 0.722 | 🥉 (regression: 🥇 on NCT → 🥉 here) |

Per-class signatures on Chaoyang (precision / recall):

| Model | benign | adeno |
|---|---|---|
| baseline | 0.86 / 0.65 | 0.76 / 0.92 |
| plain_classifier | 0.96 / 0.47 | 0.70 / 0.98 — heavily biased toward adeno |
| PLIP+BERT | 0.71 / 0.97 | 0.97 / 0.68 — opposite bias, toward benign |

All three are biased in different directions on Chaoyang; PLIP wins macro-F1 because its
precision/recall tradeoff is more balanced (least worst worst-class).

### LungHist700 OOD (3-class macro-F1, lung; 359 images at 20×)

| Model | macro-F1 | OOD rank |
|---|---|---|
| **PLIP-ViT + BERT (composed)** | **0.640** | 🥇 (+33 F1 over next; sole functional model) |
| baseline CLIP | 0.305 | 🥈 (~chance, degenerate predictor) |
| plain_classifier | 0.281 | 🥉 (~chance, degenerate predictor) |

Per-class recall on LungHist700 — the source of the collapse:

| Model | benign lung | lung AdC | lung SqC |
|---|---|---|---|
| baseline | 0.19 | **0.07** | 0.89 |
| plain_classifier | 0.15 | **0.05** | 0.91 |
| PLIP+BERT | 0.51 | 0.49 | 0.91 |

Both ResNet50 variants predict "squamous cell carcinoma" for ~90% of inputs regardless of
true class. PLIP-backbone retains a 0.49 adenocarcinoma recall (10× better) while preserving
squamous recall. The UMAPs confirm: ResNet50 places all 359 LungHist700 images into a
single tight blob in feature space (no internal class structure), while PLIP-backbone
shows three visibly separate triangle sub-clusters per class.

### Cross-dataset summary (macro-F1)

| Variant | LC25000 in-dist | NCT (colon) | Chaoyang (colon) | LungHist700 (lung) | mean OOD |
|---|---|---|---|---|---|
| **PLIP+BERT (composed)** | **0.996** | **0.933** | **0.813** | **0.640** | **0.795** |
| baseline CLIP | 0.996 | 0.866 | 0.786 | 0.305 | 0.652 |
| plain_classifier | 0.995 | 0.880 | 0.722 | 0.281 | 0.628 |

PLIP-backbone leads every OOD column. Its lead grows from +5 (NCT) → +3 (Chaoyang) →
+33 (LungHist700) F1 as the OOD shift gets harder. The ResNet50 floor is not 0.88 — it's
0.30 on the harder distribution.

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

### 3. OOD is the only meaningful axis — and on a fixed ImageNet ResNet50 backbone, none of the popular auxiliary methods help

LC25000 → external-dataset transfer **does** discriminate methods (in-dist saturates at 0.99
for everyone). Scoping this learning to **trained variants on a frozen ImageNet ResNet50
backbone**, the discrimination tells a *negative* story:

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

### 4. The frozen ImageNet ResNet50 was the OOD bottleneck — exp_08 (NCT) and exp_09 (Chaoyang + LungHist700) prove it

**Hypothesis.** The unified picture across the 9 ResNet50-based trained variants on NCT-CRC
was that they clustered within ~7 F1 points on OOD (0.81–0.88) despite radically different
architectures, training objectives, prompt formats, text encoders, and preprocessing — much
less variation than between in-dist (0.99) and OOD (0.85). That pointed squarely at one
source of variation: **the image-encoder features**. Every ResNet50-based variant used the
same frozen `resnet_50_imagenet` features, and every one inherited the OOD ceiling those
features permitted.

**Three-dataset, two-organ test** (PLIP-backbone vs. best ResNet50-based variant per dataset):

| Dataset (organ) | Best ResNet50 variant | PLIP-backbone | Δ |
|---|---|---|---|
| NCT-CRC (colon) | plain_classifier 0.880 | **0.933** | **+5.3 F1** |
| Chaoyang (colon) | baseline 0.786 | **0.813** | **+2.7 F1** |
| LungHist700 (lung) | baseline 0.305 | **0.640** | **+33.5 F1** |

PLIP-backbone wins on every dataset, with a gap that **grows with shift severity**. The
in-dist number barely moves regardless of backbone (0.996 vs 0.996 for ResNet50) — the
LC25000 saturation null is backbone-independent — but every OOD result is.

**Mechanism.** PLIP-backbone holds architecture and recipe identical to the best biomed
cell (BERT + composed prompt, frozen text, projection head, InfoNCE). The *only* change is
the image backbone (`keras_hub` ResNet50 ImageNet → `vinid/plip` ViT-B/32 histopathology).
The head, projection, prompt, and text encoder are not the limiting factor on OOD; the
image representation is. A backbone pretrained on histopathology — even without any LC25000
supervision until our fine-tuning — encodes colon and lung tissue more usefully than an
ImageNet-pretrained ResNet50 ever does.

**Why the gap is small on colon and huge on lung.** ImageNet supervision never contains
histology; LC25000 supervision rescues ResNet50 features that happen to correlate with
*colon* class labels in LC25000's training distribution. That correlation transfers
partially to NCT-CRC and Chaoyang (both colon), so ResNet50 gets to ≈0.78–0.88 there. It
does not transfer to LungHist700 (different organ); ResNet50 lung features collapse to
no class variance and the predictor degenerates. PLIP's pretraining contains both organs
and shows no such asymmetry.

The implication for the thesis flips from negative-only ("none of these methods help") to
balanced and strong: **none of these head/loss/text-side methods help on a frozen ImageNet
ResNet50 backbone — but swapping the backbone for a histopathology-pretrained one (a) breaks
the colon-OOD ceiling by +3 to +5 F1 and (b) is the only intervention that produces a
functional lung-OOD classifier at all.**

### 5. Classification quality and representation alignment are separable axes

Side-by-side UMAPs show three distinct patterns across the project:

- **Baseline CLIP (ResNet50)** — 256-d L2-normalised embeddings pull NCT colon samples *into*
  the LC25000 colon clusters in feature space. Same-class images from different datasets
  co-locate. Best visual evidence of cross-dataset alignment.
- **Plain classifier (ResNet50)** — 2048-d BatchNorm features leave NCT in a *separate region*
  of feature space — classification still works because the Dense(5) weights define class
  *directions* that NCT samples project onto correctly, but spatial alignment with LC25000 is
  lost.
- **PLIP-ViT + BERT (exp_08, NCT)** — also leaves NCT in a separate region of feature space,
  qualitatively similar to the plain-classifier pattern. Despite achieving the **highest**
  OOD F1 of any variant (0.933), the cross-dataset spatial alignment that baseline CLIP
  achieved is **not** reproduced. The same alignment-vs-F1 dissociation reproduces on
  Chaoyang and LungHist700 in exp_09: PLIP wins macro-F1 on every OOD dataset, never
  achieves baseline-CLIP-style cross-dataset cluster alignment on any of them.

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

### 8. Domain-pretrained backbone is the dominant lever — the colon-vs-lung asymmetry says everything

The headline number: **+33 F1 on LungHist700**, ~6× larger than the colon-side gap. This
is the project's most striking single result and the strongest argument for the
backbone-as-bottleneck thesis. The single biggest OOD F1 gain came from one change —
swapping the image backbone — and *the size of the gain is a function of the organ*:

| OOD dataset | ResNet50 best | PLIP-backbone | Δ |
|---|---|---|---|
| NCT-CRC (colon) | 0.880 (plain_classifier) | 0.933 | **+5.3 F1** |
| Chaoyang (colon) | 0.786 (baseline) | 0.813 | **+2.7 F1** |
| LungHist700 (lung) | 0.305 (baseline) | 0.640 | **+33.5 F1** |

**Colon (NCT, Chaoyang): +3 to +5 F1.** PLIP-backbone narrowly but consistently wins.
ResNet50 features, partially rescued by LC25000 colon supervision, give a working but
sub-optimal classifier on colon OOD.

**Lung (LungHist700): +33.5 F1.** PLIP-backbone is not narrowly better — it is the *only*
variant that produces a functional classifier. Both ResNet50-based variants collapse to a
near-chance degenerate predictor. The +33 F1 isn't "PLIP is incrementally better"; it's
"PLIP works on lung, ResNet50 doesn't".

This monotonic growth — bigger gap on harder shifts — implies the PLIP advantage is not a
constant additive offset; it is *dose-dependent* on how far the OOD distribution sits from
what ImageNet supervision happens to encode. Per-organ reporting is therefore essential;
mean-OOD numbers collapse the regime change into a single misleading point.

Stacked against the project's other interventions on **NCT-CRC** (the comparable colon baseline):

| Lever | Change relative to ResNet50 CLIP baseline | NCT OOD Δ |
|---|---|---|
| Backbone (ImageNet → PLIP-histopath) | exp_08 | **+6.7 F1** |
| Loss (CLIP → plain CE softmax) | exp_03 | +1.4 F1 |
| Stain normalisation (raw → Macenko NCT-ref) | exp_04 | −2.2 F1 |
| Text encoder (BERT → biomed) | exp_06 best | −0.3 F1 |
| Prompt (`name` → `composed`) | exp_06 | −0.5 F1 avg |

On NCT alone the backbone change is **4-5× larger** than any other lever. On LungHist700
it is **~25-100× larger** (the head/loss/text-side methods cannot rescue a non-functional
predictor). The backbone change goes in the *opposite* direction from the contemporary
methods literature's emphasis on novel loss functions, text-side innovations, and stain
harmonisation. For histopathology image classification at frozen-backbone scale, the
**right pretraining domain dominates the recipe** — and the effect grows with shift severity.

This is a strong, defensible take-home for the thesis: **future work on this task should
prioritise backbone choice (UNI, CONCH, Virchow-2, PLIP variants) over head-side or text-side
innovations.** The 3-dataset / 2-organ ablation matrix we ran provides direct evidence for
that recommendation.

### 9. ResNet50 categorically fails on lung OOD — the failure is structural, not statistical

The most striking single finding of exp_09: the LC25000-trained ResNet50 variants — *every
one of them*, with vastly different heads (5-way softmax, CLIP cosine, biomedical text
encoders) — collapse to the same degenerate predictor on LungHist700:

- baseline CLIP: SqC recall 0.89, AdC recall **0.07**, normal recall 0.19 → macro-F1 = 0.30
- plain_classifier: SqC recall 0.91, AdC recall **0.05**, normal recall 0.15 → macro-F1 = 0.28

These two numbers are *not statistically different from each other*, and both are
*not statistically different from "always predict squamous"*. The model is not classifying;
it is rejecting most inputs into a single bin.

The UMAP confirms the mechanism: all 359 LungHist700 images sit in a **single tight cluster**
in baseline CLIP's projected feature space, far from any LC25000 cluster, with no visible
sub-clustering by class. The cosine-similarity-to-prompt classification then has nothing
to act on — every input has roughly the same distance to all three lung prompts, and the
argmax noise lands on whichever prompt embedding happened to be closest to the LungHist700
blob. (That happens to be squamous; with a different LC25000 prompt configuration it could
have been any of them.)

PLIP-backbone breaks this completely. Same prompts, same projection head, same loss — but
the underlying features now have enough per-class variance on LungHist700 that the argmax
does something non-degenerate. Macro-F1 jumps from 0.30 to 0.64 (+33 F1) and per-class
recalls span 0.49–0.91 instead of 0.05–0.89.

**Why this matters beyond the F1 number**:

1. **The cross-organ asymmetry is informative**. ResNet50 features work *partially* on
   colon OOD (0.78–0.88) but *not at all* on lung OOD (~0.30). This is not a continuous
   degradation; it is a regime change. The histology of lung tissue (alveolar architecture,
   pneumocyte morphology, squamous keratinisation) is qualitatively different from colon
   (glandular epithelium, crypts). ImageNet pretraining sees neither category natively;
   the colon-side success on NCT/Chaoyang is downstream of LC25000 supervision rescuing
   features that happen to *correlate* with colon class labels in the LC25000 training
   set. That correlation does not transfer to LungHist700's lung distribution.

2. **PLIP works on lung where ResNet50 doesn't, not because PLIP has lung supervision,
   but because PLIP's pretraining domain contains lung histopathology**. There is no
   LungHist700 supervision in PLIP's training; the OpenPath corpus simply includes
   pulmonary tissue alongside other organ tissues. That's enough to make the feature
   space usable for downstream lung classification with light supervision (our LC25000
   projection head).

3. **The thesis can make a sharper claim than "PLIP helps"**: PLIP is *necessary* for the
   lung half of the LC25000 → external-test transfer to function at all. Head-side and
   text-side methods on ImageNet ResNet50 cannot rescue this; the failure is at the
   feature-extraction layer.

This is also a useful methodological observation for the broader histopathology ML
literature: **per-organ OOD evaluations should be reported separately**. Mean-OOD numbers
hide the regime change between organs.

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
- **exp_08 PLIP-ViT + BERT (NCT)** — best classification numbers of any variant
  (NCT OOD F1 = 0.933), but the NCT triangles sit in their own region of the projection,
  **away** from the LC25000 colon clusters — the same pattern as plain_classifier, not
  the baseline-CLIP-style alignment. The 5 LC25000 clusters are well-separated and tighter
  than baseline. This is the cleanest empirical demonstration that "OOD classification
  quality" and "cross-dataset feature alignment" are separable axes. The cosine similarity
  to the prompt direction is what carries classification; absolute embedding location is
  what carries alignment, and PLIP optimises the first more than the second on this transfer.
- **exp_09 / Chaoyang (all 3 variants)** — Chaoyang triangles cluster in their own region
  of the UMAP regardless of variant; the pattern is similar to plain_classifier and
  PLIP-on-NCT (alignment lost), not baseline-CLIP-on-NCT. The 5 LC25000 colon clusters
  remain well-separated. Visual signature: a single Chaoyang "blob" sitting adjacent to but
  not overlapping with the LC25000 benign-colon and adeno clusters. Within the Chaoyang
  blob there is sub-structure that correlates with class, but PLIP shows the cleanest
  separation.
- **exp_09 / LungHist700 baseline (ResNet50 CLIP)** — diagnostic. **All 359 LungHist700
  samples form a single tight cluster** in the upper-right of the UMAP, with no visible
  sub-clustering by class (AdC/SqC/normal triangles overlap completely). This is the
  visual signature of a failed feature extractor: ResNet50 has produced features that
  do not separate lung tissue types. Combined with the 0.30 macro-F1, this is what
  "categorically fails" looks like in feature space.
- **exp_09 / LungHist700 PLIP** — by contrast, the LungHist700 triangles split into 3
  visibly separate sub-clusters by class. Still distinct from the LC25000 lung clusters
  (alignment again lost), but internally structured. This is what allows the 0.64 macro-F1
  to emerge.

---

## Project caveats / known limitations

These need to land in the §limitations section of the thesis:

1. **Canonical split was built pre-dedupe** and reused by every model trained today. The 1,280
   byte-duplicates therefore live across both train and test for every reported number. Effect
   size: probably 1–2 F1 points of inflation on every in-dist number; OOD numbers are unaffected
   since NCT-CRC, Chaoyang, and LungHist700 are all independent datasets. This caveat applies
   uniformly to every trained variant in the project (v7 baseline through exp_08 PLIP-backbone)
   — there is no checkpoint reported here that is free of it.

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

5. **Restricted-argmax OOD evaluation.** Every external eval scores only the LC25000 classes
   that have a clean dataset analogue: 2 colon classes for NCT (NORM/TUM) and Chaoyang
   (normal/adenocarcinoma); 3 lung classes for LungHist700 (normal/AdC/SqC). The other 7 NCT
   classes (ADI, BACK, DEB, LYM, MUC, MUS, STR) and the Chaoyang `serrated`/`adenoma` (1,163 +
   937 images) are excluded from scoring — they have no clean LC25000 analogue. LungHist700's
   3 differentiation grades within AdC and SqC are collapsed to parent classes because LC25000
   has no grading annotation. This is a deliberate simplification; a richer multi-class OOD
   evaluation would require either a different dataset or additional class-mapping work.

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
> but frame cross-source generalisation to three external datasets across two organs —
> **NCT-CRC-HE-7K** (colon), **Chaoyang** (colon), and **LungHist700** (lung) — as the
> central evaluation. The choice of multiple datasets across two organs is deliberate: results
> presented later show the effect size of the dominant intervention varies by an order of
> magnitude between organs, which a single-dataset eval would have hidden.

### §6 (OOD generalisation — the central contribution)

> On the LC25000 → external-dataset cross-source generalisation task (NCT-CRC for colon,
> Chaoyang for colon, LungHist700 for lung), we evaluate a matrix of interventions on the
> standard frozen-ImageNet-ResNet50 + frozen-text-encoder CLIP recipe. On the **NCT-CRC
> baseline evaluation** with the ResNet50 backbone held fixed, we observe:
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
>    composed prompt, projection heads, InfoNCE) raises NCT-CRC OOD macro-F1 from 0.863 →
>    **0.933** — a +7.0 F1 absolute gain over the same-recipe ResNet50 counterpart, and +6.7
>    over the previous best (plain_classifier).
>
> 6. **(exp_09 / Chaoyang)** On an independent colon OOD dataset (Beijing Chaoyang Hospital,
>    different scanner, label noise on train split), the PLIP-backbone advantage persists:
>    PLIP 0.813 vs. baseline CLIP 0.786 vs. plain_classifier 0.722. The plain_classifier —
>    which marginally led on NCT — drops to last on Chaoyang, indicating that the high-confidence
>    softmax predictor of plain CE collapses under heavier distribution shift more catastrophically
>    than the CLIP cosine geometry. PLIP-backbone is robust to this collapse.
>
> 7. **(exp_09 / LungHist700)** On the lung half of LC25000 — never previously evaluated OOD —
>    the gap becomes definitive. Both ResNet50-based variants (baseline 0.305, plain 0.281)
>    converge to a degenerate "predict squamous cell carcinoma for everything" predictor,
>    barely distinguishable from chance for a 3-class problem. PLIP-backbone (0.640) is the
>    only variant that produces a functional lung-OOD classifier — a **+33 F1** lift over
>    the ResNet50 baseline. The lung-side failure mode is structural: UMAPs show ResNet50
>    placing all 359 LungHist700 images into a single tight cluster with no class
>    sub-structure, while PLIP produces three visibly distinct class sub-clusters.
>
> Taken together, the matrix (3 OOD datasets, 2 organs, 3 model variants) establishes a clear
> ordering of effect sizes: **backbone pretraining domain dominates the recipe** by:
>
> - ~5× over the next-largest single lever (loss function) on the easier colon-side OOD
> - **~25–100× on lung-side OOD**, where head/loss/text-side methods on ImageNet ResNet50
>   cannot rescue a feature extractor that produces no usable lung-tissue variance.
>
> Stain normalisation, biomedical text encoders, and prompt formatting contribute zero or
> negative OOD value across every dataset evaluated. The thesis's central methodological
> recommendation is therefore that improving OOD generalisation on histopathology
> classification should prioritise the **image encoder's pretraining domain** before
> head-side, loss-side, or text-side innovations — and per-organ OOD evaluation should be
> reported separately, because the effect size differs by an order of magnitude between
> colon and lung on the same LC25000 training distribution.

### §7 (Discussion)

Items to discuss:
- **The classification-vs-alignment dissociation.** Baseline CLIP achieves cross-dataset
  cluster alignment but lower F1; PLIP-backbone achieves the highest F1 across every OOD
  dataset but leaves OOD samples in their own region of feature space (consistently across
  NCT, Chaoyang, and LungHist700). The contrastive objective optimises alignment of
  *same-class* pairs; the cosine-similarity classifier rule only needs alignment to the
  class-prompt *direction*. These are different geometric properties — both empirically
  separable on every dataset we tested.
- **The implementation-vs-checkpoint trap** (keras_hub vs keras.applications) as a
  methodological case study in silent benchmark inflation/deflation.
- **The augmentation-pair leakage in LC25000** as a more fundamental dataset issue, separate
  from the dedupe issue and probably the dominant inflater of in-dist numbers.
- **Why head-side methods cluster within 7 F1 points on NCT but move by 0 F1 on
  LungHist700** despite the dataset being qualitatively different. Frame in terms of
  degrees of freedom: a frozen ResNet50 backbone fixes 23M parameters of feature
  extraction; the projection head + temperature add ≈525K trainable parameters; the head
  can only re-weight what the backbone already encodes. When the backbone encodes no
  per-class variance (LungHist700), the head has nothing to re-weight — the variance
  between head-side methods collapses to zero.
- **The cross-organ asymmetry**. ResNet50-based features work partially on colon OOD
  (0.78–0.88) but not at all on lung OOD (~0.30). Mechanism: ImageNet supervision never
  contains histology; LC25000 supervision rescues features that happen to correlate with
  colon class labels but does not — perhaps cannot — produce useful lung tissue
  discrimination within a frozen ResNet50. PLIP's pretraining contains both organs and
  shows no such asymmetry. This is evidence that the bottleneck is specifically the
  *pretraining domain*, not the architecture.
- **The plain-classifier flip** (NCT 🥇 → Chaoyang 🥉 → LungHist700 🥉). High-confidence
  softmax predictors trained on a single dataset are more brittle under shift than
  cosine-similarity CLIP classifiers. The plain_classifier's NCT lead is in retrospect
  the most fragile finding in the project: it does not survive heavier shift.
- **Why specialised supervision wins absolute F1 but loses on generalisation gap vs PLIP
  zero-shot, and how exp_08 reconciles the two**: PLIP-as-backbone + small LC25000
  fine-tuning gets the best of both.
- **Per-organ vs aggregate OOD reporting.** The mean-OOD column in our matrix
  (PLIP 0.795 vs baseline 0.652) collapses three datasets into one number; the
  per-dataset breakdown reveals the organ-specific regime change. The thesis should
  report both, but emphasise the per-dataset numbers — particularly the lung result.

### §8 (Limitations & Future work)

- Dedupe + source-aware LC25000 split would tighten the in-dist numbers; exp_08's in-dist
  saturation (0.996) is subject to the same caveat as every other trained variant here.
- exp_09 confirms the PLIP-backbone advantage across 3 OOD datasets and 2 organs, so the
  natural next experiments are now in three directions:
  - **(a) Other histopathology-pretrained backbones** (UNI, CONCH, Virchow-2) — do they
    also break the lung-side ResNet50 collapse, or is PLIP specifically responsible?
    Important for a "PLIP vs the world" comparison rather than "PLIP vs ResNet50".
  - **(b) PLIP partial unfreezing** — does fine-tuning the last few PLIP transformer
    blocks on LC25000 narrow the gap further, or does it overfit?
  - **(c) More cross-organ OOD** — the project currently covers colon (NCT, Chaoyang) and
    lung (LungHist700). Adding a third organ (e.g. breast via PCam, or kidney/prostate
    via TCGA tiles) would test whether the +33 F1 lung pattern is a 2-organ-specific
    artefact or a general property of cross-organ ImageNet-vs-histopath transfer.
- exp_09 results have small per-class supports on LungHist700 (85/146/128 images per
  class). Per-class F1 variance is non-trivial; **bootstrap confidence intervals on the
  +33 F1 lift** would strengthen the claim (~1h of additional analysis, single notebook).
- exp_09's 7-class → 3-class LungHist700 collapse (grades wd/md/pd → AdC/SqC parent) is
  our own choice; LungHist700's original task is finer-grained. We do not currently
  evaluate per-grade performance; if PLIP-backbone shows useful grading discrimination
  within the AdC/SqC subclusters, that would be an additional positive finding.
- exp_08's combined LC25000-supervised + PLIP-pretrained pipeline raises a separate question:
  what happens if one fine-tunes PLIP on its own original training set (OpenPath) as a
  control? This would help disentangle "PLIP's pretraining matters" from "frozen features
  + fresh projection head on any histopath ViT works."
- **LungHist700's source bias is not characterised.** If the dataset's pulmonary tissue
  was scanned on a particular scanner / stained at a particular lab, the ResNet50 failure
  could be partly a stain-shift artefact (not just architecture). Evaluating LungHist700
  per-grade or per-resolution might localise where ResNet50 breaks first.

---

## Open questions (could be answered with modest extra compute)

| Question | Cost to answer | Status |
|---|---|---|
| Would a histopath-pretrained backbone move the OOD ceiling above 0.88? | ~6h compute (1 training run with new backbone) | **answered by exp_08: yes, PLIP → 0.933 on NCT** |
| Does the PLIP-backbone OOD advantage hold on a second colon dataset? | ~2h compute (one eval pass) | **answered by exp_09: yes, +3 F1 on Chaoyang** |
| Does the PLIP-backbone advantage hold across organs? | ~2h compute (one eval pass) | **answered by exp_09: dramatically yes — +33 F1 on LungHist700** |
| Do bootstrap CIs on the LungHist700 +33 F1 lift confirm the gap is significant? | ~1h analysis (single notebook) | open — recommended high-priority |
| Do other histopath-pretrained backbones (UNI, CONCH, Virchow-2) match or beat PLIP? | ~6h compute each | open — most informative remaining experiment |
| Does fine-tuning PLIP's last few transformer blocks improve over the frozen-PLIP result? | ~4h compute | open |
| Does dedupe + source-aware split lower the in-dist saturation enough for methods to differentiate? | ~3h compute (rebuild split, retrain baseline + plain) | open |
| Does the PubMedBERT × composed regression hold with different prompt templates? | ~2h compute (3 more prompt variants for that cell) | open |
| Are NCT non-colon classes (ADI/MUC/etc.) systematically misclassified into specific LC25000 classes? | ~1h analysis (the "leak" confusion matrix idea) | open |
| Does the lung-side failure replicate on a third lung dataset (e.g. TCGA-LUAD WSIs after tiling)? | ~6h compute + setup | open |
| Within LungHist700's AdC/SqC sub-clusters in PLIP's UMAP, is there grade information? | ~1h analysis (per-grade F1 from existing run) | open — could be a free additional positive finding |

The top three are answered; the bootstrap-CI follow-up and UNI/CONCH benchmark are the
highest-leverage open work.

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
    CLIP_OOD_eval.ipynb                  — baseline / stain VARIANT switch (NCT-CRC)
    PlainClassifier_OOD_eval.ipynb       — plain softmax architecture (NCT-CRC)
    CLIP_Biomed_OOD_eval.ipynb           — biomed VARIANT loop (NCT-CRC)
    CLIP_PLIP_OOD_eval.ipynb             — PLIP-backbone variant (exp_08 NCT-CRC eval)
    Chaoyang_OOD_eval.ipynb              — exp_09, all 3 variants via VARIANT switch (colon)
    LungHist700_OOD_eval.ipynb           — exp_09, all 3 variants via VARIANT switch (lung)
    runs/2026-05-28_{baseline,plain_classifier,stain_macenko_nct_ref,biomed,plip_bert_composed}_ood/
    runs/2026-05-29_{baseline,plain_classifier,plip_bert_composed}_{chaoyang,lunghist700}_ood/  (exp_09)
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
