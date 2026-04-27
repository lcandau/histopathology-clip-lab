# histopathology-clip-lab
Medical histopathology images AI lab using CLIP model for semi-supervised training. The aim of this repository is to explore the CLIP model for histopathology as a diagnosis aid.

This project investigates the use of vision-language models such as CLIP for zero-shot and weakly supervised classification of histopathology images, aiming to assess their potential as diagnostic support tools under limited annotation scenarios.

---

# 🧬 Histopathology CLIP Lab (ResNet50 - Baseline)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)  
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)  
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔬 Project Overview

This repository explores the use of **CLIP-style multimodal models** for **histopathology image understanding**, focusing on **zero-shot classification**.

Instead of relying on large annotated datasets (which are scarce in medical domains), this project investigates whether a model can:

- Learn **joint representations of images and text**
- Use **natural language prompts** (e.g., *"A histopathology image of lung adenocarcinoma"*)
- Perform **zero-shot classification** without task-specific training

### 🧠 Core Idea

We align:
- 🖼️ Image embeddings (via ResNet50)
- 📝 Text embeddings (via BERT)

…into a shared latent space using **contrastive learning**, enabling inference via similarity.

---

## 🧠 Methodology

### 🏗️ Model Architecture

- **Image Encoder**
  - ResNet50 (ImageNet pretrained)
- **Text Encoder**
  - BERT tokenizer + encoder
- **Projection Layers**
  - Map both modalities into a shared embedding space
- **Training Objective**
  - Contrastive loss (CLIP-style)

---

### ⚙️ Training Strategy

Each image is paired with a descriptive prompt:

"A histopathology image of lung adenocarcinoma"

The model learns to:
- Maximize similarity for correct image-text pairs
- Minimize similarity for incorrect pairs

---

### 🔍 Zero-Shot Evaluation

At inference time:

1. Generate embeddings for candidate text prompts  
2. Compare them with image embeddings  
3. Select the class with highest similarity  

➡️ No classifier head required

---

### 🗂️ Dataset

- Histopathology images:
  - Lung (normal, adenocarcinoma, squamous cell carcinoma)
  - Colon (normal, adenocarcinoma)
- RGB images
- Classification task (no segmentation masks)

---

### 📊 Evaluation Methods

- Confusion Matrix  
- Cosine Similarity Scoring  
- Embedding Visualization (UMAP)  
- Zero-shot classification accuracy  

---

## ⚠️ Limitations & Challenges

### 1. 📉 Limited Dataset Size
- Small and imbalanced medical dataset  
- Impacts generalization  

**Mitigation:**  
- Zero-shot learning reduces dependence on labels  

---

### 2. 🌍 Domain Gap (ImageNet → Histopathology)
- Pretrained on natural images  
- Histopathology has very different visual patterns  

⚠️ Still unresolved  

---

### 3. 📝 Prompt Sensitivity
- Predictions depend heavily on wording  

Example:
- "lung cancer" vs "lung adenocarcinoma"  

**Mitigation:**  
- Standardized prompt templates  
- Iterative prompt refinement  

---

### 4. ⚙️ Training Instability
- Contrastive learning sensitive to:
  - Batch size  
  - Normalization  
  - Loss scaling  

**Mitigation:**  
- Progressive improvements across versions  

---

### 5. 📊 Evaluation Complexity
- Harder than standard classification  

**Mitigation:**  
- Added visualization + similarity-based evaluation  

---

## 📦 Version History

### 🔹 v1 — Initial CLIP Prototype
- CLIP-style architecture (ResNet50 + BERT)  
- Prompt-based training  
- Basic contrastive loss  
- Initial dataset pipeline  

---

### 🔹 v2 — Zero-Shot Pipeline
- Introduced zero-shot classification  
- Prompt-based inference  
- Improved evaluation structure  

---

### 🔹 v3 — Evaluation Improvements
- Added confusion matrix  
- Embedding similarity analysis  
- More robust metrics  

---

### 🔹 v4 — Pipeline Refactor
- Cleaner training/evaluation separation  
- Improved preprocessing  
- Better reproducibility  

---

### 🔹 v5 — Generalization Improvements
- Data augmentation  
- Improved robustness  
- Better zero-shot performance  

---

### 🔹 v6 — Final Experimental Version
- Consolidated pipeline  
- Clean notebook structure  
- Ready for reporting / thesis usage  

---

## 🚀 Future Work

- Fine-tune backbone on histopathology datasets  
- Explore **ViT-based CLIP models**  
- Multi-prompt ensembling  
- Few-shot learning benchmarks  
- Extend to weak segmentation tasks  

---

## 🧾 Summary

This project shows that:

- CLIP-style models can be adapted to **medical imaging**  
- Zero-shot classification is **feasible but limited**  
- Performance is constrained by:
  - Data availability  
  - Domain mismatch  
  - Prompt design  

➡️ Provides a foundation for **low-supervision medical AI systems**

---

## 👨‍🔬 Author

**Leandro Candau**  
AI & Software Engineer  

---

## 📄 License

MIT License
