# 🧬 Histopathology CLIP Lab

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)  
![Status](https://img.shields.io/badge/Status-Research%20Project-yellow)  

---

## 🔬 Project Overview

This project explores the application of **Foundational Models (FM)** to **oncology histopathology imaging**, with a focus on **self-supervised and multimodal learning**.

Modern foundational models aim to learn **general representations from raw data without explicit labels**, enabling strong performance on downstream tasks such as classification, anomaly detection, clustering, and segmentation. This repository serves as a sandbox to evaluate how well these architectures adapt to the complex, highly specific domain of cellular pathology.

---

## 🧠 Research Objective

The main goal of this project is to evaluate, tune, and compare different foundational model approaches on histopathology data:

### 1. CLIP (Multimodal Learning)
- Align image (tissue patches) and text representations (diagnostic prompts).
- Evaluate **zero-shot classification** capabilities in a strictly medical context.
- Measure **image-label consistency** and retrieval performance (Recall@K).

### 2. ViT-based Models
- Replace standard CNN backbones with Vision Transformers.
- Evaluate structural representation quality and attention mapping.

### 3. DINOv2 (Self-Supervised Learning)
- Learn dense visual features entirely without labels.
- Leverage resulting embeddings for:
  - Clustering (UMAP / k-Means)
  - Anomaly detection
  - Downstream linear-probing tasks

---

## 📁 Repository Structure

The project is structured to isolate distinct experiments while sharing common utilities.

    histopathology-clip-lab/
    │
    ├── data/                       # Raw and processed dataset files (ignored in git)
    │
    ├── experiments/                # Isolated experiment environments
    │   │
    │   ├── exp_01_baseline/        # 🔹 CLIP with ResNet50 + BERT
    │   │   ├── notebooks/          # Iterative Jupyter notebooks (v1 -> v6+)
    │   │   ├── checkpoints/        # Saved weights (.h5), splits, and history logs
    │   │   └── README.md           # Experiment-specific documentation & changelogs
    │   │
    │   ├── exp_02_vit/             # 🔹 CLIP with Vision Transformer backbone
    │   │   └── README.md
    │   │
    │   ├── exp_03_dino/            # 🔹 CLIP + DINO-pretrained ViT backbone
    │   │   └── README.md
    │   │
    ├── src/                        # Shared utilities (if abstracted from notebooks)
    │   ├── data_loader.py
    │   └── evaluation.py
    │
    ├── README.md                   # Project overview (You are here)
    └── requirements.txt            # Shared dependencies (keras-hub, tensorflow, etc.)

---

## 🧪 Experimental Roadmap

| Experiment | Status | Description |
|----------|:---:|------------|
| [**exp_01_baseline**](experiments/exp_01_baseline/README.md) | 🟢 Active | Baseline CLIP with ResNet50 (vision) and BERT (text). Focus on prompt ensembling, contrastive loss, and baseline zero-shot metrics. |
| [**exp_02_vit**](experiments/exp_02_vit/README.md) | ⚪ Planned | Replacing ResNet50 with a Vision Transformer backbone trained from scratch to evaluate representational improvements. |
| [**exp_03_dino**](experiments/exp_03_dino/README.md) | ⚪ Planned | Hybrid approach using a DINOv2 self-supervised pretrained ViT as the image encoder inside the CLIP framework. |

Each experiment deeply explores:
- Representation learning quality  
- Zero-shot classification & F1 performance  
- Embedding structure (via UMAP visualizations)  
- Generalization capabilities across varying stains/patients

---

## 📊 Dataset

This project utilizes the `andrewmvd/lung-and-colon-cancer-histopathological-images` dataset from Kaggle, which includes:
- **5 distinct classes:** Lung benign, Lung adenocarcinoma, Lung squamous cell carcinoma, Colon benign, Colon adenocarcinoma.
- **Format:** RGB image patches (224x224).
- **Labels:** Dynamically ensembled into descriptive text prompts during training.

---

## 🔍 Key Questions

This project aims to answer:
1. Can CLIP perform highly accurate **zero-shot classification** in medical imaging without fine-tuning on massive medical datasets?
2. How much does the **backbone architecture** (CNN vs. ViT) impact cross-modal alignment?
3. Can **self-supervised models (DINO)** fundamentally outperform supervised multimodal approaches by learning deeper tissue semantics?
4. Do the resulting high-dimensional embeddings organically reveal **hidden structures** (e.g., morphological subclusters or slide anomalies)?

---

## ⚠️ Challenges

- **Domain Gap:** Transitioning models pre-trained on natural images (ImageNet/WebImage) to histopathology.
- **Prompt Sensitivity:** Contrastive learning performance is highly sensitive to the phrasing of diagnostic text prompts.
- **Hardware Limitations:** Managing large effective batch sizes required for InfoNCE loss on limited GPU memory.
- **Stain Variance:** Accounting for H&E dye inconsistencies across different laboratory slides.

---

## 👨‍🔬 Author

**Leandro Candau** Seniour Software Engineer | Student for Applied Machine Learning Engineer

---

## 📄 License

MIT License