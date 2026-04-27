# 🧪 Experiment 01 — CLIP Baseline (ResNet50 + BERT)

Welcome to the baseline experiment for multimodal contrastive learning applied to histopathology images. 

## 🎯 Objective

Establish a **baseline multimodal model** using CLIP to evaluate:
- Image-text alignment
- Zero-shot classification performance
- Embedding quality in histopathology data

*This serves as the **reference experiment** for all subsequent architectural and data improvements.*

---

## 🧠 Methodology

### Architecture
- **Image Encoder:** ResNet50 *(ImageNet pretrained)*
- **Text Encoder:** BERT
- **Projection Layers:** Shared embedding space
- **Loss:** Contrastive loss (InfoNCE)

### Training Setup
During training, each image is paired with a dynamically generated prompt template, for example:
> `"A histopathology image of [class]."`

**The model is trained to learn:**
* ✅ High cosine similarity for correct image-text pairs.
* ❌ Low cosine similarity for incorrect image-text pairs.

### Inference (Zero-Shot)
1. Encode all candidate textual prompts.
2. Compare them against the target image embedding.
3. Select the prompt with the highest similarity score.

---

## 🗂️ Dataset

This experiment uses lung & colon histopathology images, categorized into **5 distinct classes**:
1. Lung normal (benign tissue)
2. Lung adenocarcinoma
3. Lung squamous cell carcinoma
4. Colon normal (benign tissue)
5. Colon adenocarcinoma

---

## 📊 Evaluation Metrics

To thoroughly assess the model's capabilities, we track:
* **Zero-shot Accuracy & Macro F1:** Overall classification metrics.
* **Confusion Matrix:** To identify specific inter-class misclassifications.
* **Retrieval Scores:** Cross-modal Image-to-Text and Text-to-Image recall.
* **UMAP Visualization:** 2D projections of the high-dimensional embedding space.

---

## ⚠️ Limitations

* **Domain Gap:** ResNet50 was pretrained on natural images (ImageNet), which differs vastly from cellular pathology structures.
* **Prompt Sensitivity:** The model's retrieval performance can vary based on the exact wording of the text prompts.
* **Data Constraints:** We are operating with a limited dataset size and potential class imbalances.

---

## 📈 Results (Summary)

> ⏳ *(TBD once we finalize the current training run)*

- **Zero-shot accuracy:** `XX%`
- **Macro F1 Score:** `XX%`
- **Key Observations:**
  - Strong clustering observed in: `[Class X]`
  - Notable confusion between: `[Class Y]` and `[Class Z]`

---

## 🔗 Related Experiments

- 👉 **ViT Version:** `../exp_02_vit/`
- 👉 **DINO Version:** `../exp_03_dino/`

---

## 🗺️ Roadmap & Next Steps

Our immediate goals to elevate this baseline to a state-of-the-art approach:

- [ ] **Address Contrastive "False Negatives":** With only 5 classes, standard InfoNCE loss incorrectly penalizes the model when two images of the same class appear in the same batch. We will explore **Supervised Contrastive Loss (SupCon)** or a soft-target InfoNCE loss.
- [ ] **Scale the Effective Batch Size:** Contrastive learning relies heavily on a large pool of hard negatives. We plan to implement **Gradient Accumulation** to simulate much larger batch sizes (e.g., 64 or 128) on limited hardware.
- [ ] **Upgrade to Domain-Specific Backbones:** - *Text Encoder:* Swap general BERT for **PubMedBERT** or **ClinicalBERT** to natively handle complex medical terminology.
  - *Vision Encoder:* Explore Vision Transformers (ViT) or domain-pretrained pathology encoders like **CTransPath** or **Phikon**.
- [ ] **Advanced Pathology Augmentations:** Enhance the data pipeline with domain-specific techniques like **Macenko stain normalization** and spatial transformations (e.g., random rotations, elastic deformations).
- [ ] **Parameter-Efficient Fine-Tuning (PEFT):** Experiment with **LoRA (Low-Rank Adaptation)** on the text encoder to allow it to adapt to our specific dataset without risking catastrophic forgetting.
- [ ] **Prevent Potential Data Leakage:** Restructure future train/test splits strictly at the **patient level** or **WSI (Whole Slide Image) level** to guarantee the model learns generalizable cancer features rather than patient-specific artifacts.

---

## 📦 Version History

<details>
<summary><strong>Click to expand previous versions (v1 - v5)</strong></summary>

### 🔹 v1 — Initial Baseline Setup
* **Base Models:** Initialized the dual-encoder architecture using `keras_hub` with `resnet_50_imagenet` (Vision) and `bert_base_en_uncased` (Text) backbones. Both frozen.
* **Architecture Dimensions:** `IMG_SIZE = 224`, `MAX_LEN = 32`, and `EMBED_DIM = 256`.
* **Training Hyperparameters:** `BATCH_SIZE = 4`, `10` Epochs, `Adam` optimizer (`lr = 1e-4`).
* **Dataset & Processing:** Downloaded `andrewmvd` dataset via `kagglehub`.
* **Custom CLIP Model:** Defined custom `keras.Model` implementing InfoNCE contrastive loss and trainable temperature (`logit_scale`).

### 🔹 v2 — Zero-Shot Pipeline
* **Batch Size:** Increased from `4` to `8`.
* **Epochs:** Increased from `10` to `20`.

### 🔹 v3 — Evaluation Improvements
* **Dataset Splitting:** Shifted to an 80/10/10 train/validation/test split.
* **Data Loading:** Implemented structured `CLASS_INFO` dictionaries for cleaner label management.
* **Model Checkpointing:** Added `ModelCheckpoint` to automatically save best weights.

### 🔹 v4 — Pipeline Refactor
* **State Management:** Added robust save/load logic for model weights, dataset splits (`split.json`), and training history to Google Drive.
* **Evaluation Metrics:** Added Macro F1 score evaluation.
* **Cross-Modal Retrieval:** Implemented rigorous Image-to-Text (i2t) and Text-to-Image (t2i) retrieval evaluation loops.

### 🔹 v5 — Generalization Improvements
* **Hyperparameter Tuning:** Reduced `MAX_LEN` to 16, Epochs to 10, and `LR` to `1e-5`. Added Weight Decay (`1e-4`).
* **Model Architecture:** Partially unfroze ResNet50 (last 30 layers trainable). Added Dropout (0.2) and LayerNormalization to projection heads.
* **Optimizer:** Switched to `AdamW`.
* **Data Augmentation:** Added robust training augmentations (random flips, brightness, contrast, saturation, hue).

</details>

### 🔹 v6 — (Latest) Prompt Improvements
* **Dynamic Prompt Ensembling:** Removed static prompts. Introduced a `PROMPT_TEMPLATES` list (e.g., *"A histopathology image of {}."*, *"An H&E stained tissue image showing {}."*) randomly sampled during training to improve text encoder robustness.
* **Label Encoding:** Updated extraction loops to store numerical class indices instead of strings to reduce memory overhead.
* **tf.data Pipeline Updates:** Refactored map functions to generate dynamic prompts on the fly during training, and standardized prompts during evaluation.
* **Visualization Checks:** Updated visualization helpers to dynamically map numeric labels back to human-readable text.