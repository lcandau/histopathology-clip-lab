# Histopathology CLIP Lab

Master's project (TFM): *A Study on Foundational Visual-Language Models for Histopathology Image Analysis*.

**Author:** Leandro Candau Sánchez de Ybargüen
**Tutors:** Miguel Ángel Gutiérrez Naranjo, Miguel Cárdenas Montes (co-director)
**Institution:** Master en Lógica, Computación e IA · ETSI Informática · Universidad de Sevilla
**Defense target:** 2026

---

## What this repository contains

| Path | Purpose |
| --- | --- |
| `report/` | LaTeX sources for the project report. The current PDF is `report/tfmETSI.pdf` (browsable on GitHub). |
| `experiments/exp_01_baseline/` | CLIP + ResNet50 family — baseline (frozen) and partial fine-tuning (`unfreeze30`). |
| `experiments/exp_02_plip_zeroshot/` | PLIP zero-shot evaluation on the same test split. |
| `experiments/exp_03_plain_classifier/` | No-CLIP-architecture control row: ResNet50 + Dense(5) + softmax. Frozen and `unfreeze30` variants. |
| `experiments/exp_07_image_encoder_vit/`, `experiments/exp_08_image_encoder_dino/` | Placeholders for the optional image-encoder family experiments (deferred). |
| `experiments/<exp>/runs/` | Executed notebooks with embedded outputs, one folder per dated run. |
| `src/` | Shared helpers used by every notebook (dataset constants, splits, prompt templates, reproducibility). |
| `data/splits/` | Persisted train/val/test split JSON so every experiment evaluates on identical images. |
| `results/metrics/` | Per-experiment classification metrics (JSON + CSV). |
| `results/plots/` | Loss curves, UMAP, similarity heatmaps. |
| `results/confusion_matrices/` | Confusion matrices (PNG + NPY). |
| `requirements.txt` | Pinned versions for the TensorFlow / KerasHub stack used by the baseline. |

---

## Quick links for tutor review

- **Current PDF**: [`report/tfmETSI.pdf`](report/tfmETSI.pdf)
- **Baseline notebook (template)**: [`experiments/exp_01_baseline/CLIP_ResNet50_baseline.ipynb`](experiments/exp_01_baseline/CLIP_ResNet50_baseline.ipynb)
- **PLIP zero-shot notebook**: [`experiments/exp_02_plip_zeroshot/CLIP_PLIP_zeroshot.ipynb`](experiments/exp_02_plip_zeroshot/CLIP_PLIP_zeroshot.ipynb)
- **Plain image classifier notebooks**: [`experiments/exp_03_plain_classifier/`](experiments/exp_03_plain_classifier/)
- **Executed baseline run**: [`experiments/exp_01_baseline/runs/2026-05-25_baseline_rerun/`](experiments/exp_01_baseline/runs/2026-05-25_baseline_rerun/)
- **Executed PLIP run**: [`experiments/exp_02_plip_zeroshot/runs/2026-05-24_plip_zeroshot/`](experiments/exp_02_plip_zeroshot/runs/2026-05-24_plip_zeroshot/)

---

## Experiments and status

| # | Experiment | Status | Headline result |
| --- | --- | :---: | --- |
| Baseline | CLIP with frozen ResNet50 + frozen BERT, LC25000 | done | 63.0% accuracy, 62.5% macro F1 |
| PLIP zero-shot | PLIP frozen, no LC25000 training | done | 69.8% accuracy, 70.0% macro F1 (+7.6 macro-F1 over baseline) |
| Linear probe | CE classifier on frozen ResNet50 features | pending | — |
| Partial fine-tuning | Last 30 layers of ResNet50 unfrozen | pending | — |
| Stain normalization (Macenko) | Same baseline with normalized images | pending | — |
| External colon dataset | Zero-shot transfer of baseline | pending | — |
| DINOv2 image encoder (stretch) | DINOv2 frozen + BERT | pending | — |

---

## Dataset

Kaggle [LC25000](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images): 25,000 H&E patches at 768×768, five tissue classes (benign lung, lung adenocarcinoma, lung squamous cell carcinoma, benign colon, colon adenocarcinoma). The 25,000 images derive from 1,250 originals expanded by the authors via rotation/flip augmentation — this is the reason no further geometric augmentation is applied during training.

Stratified 80/10/10 train/val/test split with seed 42, persisted to `data/splits/lc25000_seed42.json`.

---

## Running the notebooks

The notebooks are designed to run on Google Colab. Each notebook's first cell clones this repository and installs the pinned dependencies, then mounts Google Drive for weight persistence. To run a notebook:

1. Open Colab → **File → Open notebook → GitHub** → `lcandau/histopathology-clip-lab`.
2. Pick the notebook (e.g. `experiments/exp_01_baseline/CLIP_ResNet50_baseline.ipynb`).
3. Make sure the runtime is set to a T4 GPU.
4. Run all cells.

Local execution is also supported with the pinned `requirements.txt`, but Colab is the primary platform.

---

## Repository conventions

- The notebooks under `experiments/<exp>/` are clean *templates* with no embedded outputs.
- Completed runs are archived under `experiments/<exp>/runs/<YYYY-MM-DD>_<tag>/` with the executed notebook preserved.
- Canonical artifacts (metrics JSON/CSV, figure PNGs, confusion-matrix NPYs) live in `results/` so the LaTeX report can include them directly.

---

## 👨‍🔬 Author

- **Leandro Candau** Seniour Software Engineer | Student for Applied Machine Learning Engineer

---

## License

MIT.
