# Chest X-Ray Pneumonia Detection Using Convolutional Neural Networks

A comparative and explainable study of a custom CNN, VGG16, and ResNet50 for binary classification of paediatric chest X-rays into **Normal** and **Pneumonia**.

**Author:** Mani Teja Padala
**Institution:** Department of Business, University of Europe for Applied Sciences, Potsdam, Germany

---

## Overview

Pneumonia is a leading cause of death globally, particularly for children under five, and is diagnosed from chest X-rays that require scarce expert radiologists. This project trains and compares three CNN models for automated pneumonia screening, with an emphasis on **sensitivity** (the clinically critical metric, since a missed pneumonia case is the most dangerous error) and on **explainability** via Grad-CAM.

The study is deliberately comparative: a custom CNN trained from scratch is evaluated against two ImageNet-pretrained transfer-learning models (VGG16 and ResNet50) under identical, imbalance-aware conditions.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kermany et al., sourced from Guangzhou Women and Children's Medical Center.

| Attribute | Details |
|---|---|
| Total images | 5,856 JPEG chest X-rays |
| Classes | 2 (Normal, Pneumonia) |
| Class balance | ~27% Normal / ~73% Pneumonia (imbalanced) |
| Input resolution | Resized to 224 × 224, RGB |
| Train split | 5,216 (1,341 N / 3,875 P) |
| Validation split | 16 (8 N / 8 P) |
| Test split | 624 (234 N / 390 P) |

> **Note:** The image data is **not** included in this repository. Download it from the Kaggle link above and place it in a `data/` folder (see Setup).

## Methodology

- **Preprocessing:** resize to 224×224, convert to RGB, rescale pixels to [0, 1].
- **Augmentation (training set only):** rotation (±20°), width/height shift, zoom, horizontal flip, brightness adjustment.
- **Class-imbalance handling:** balanced class weights (Normal ≈ 1.945, Pneumonia ≈ 0.673).
- **Models:**
  - **Custom CNN** — four convolutional blocks (32→64→128→256), batch norm, max pooling, global average pooling, dense head with dropout, sigmoid output.
  - **VGG16** — ImageNet-pretrained base, two-stage freeze-then-fine-tune, custom classification head.
  - **ResNet50** — ImageNet-pretrained base, identical head and training procedure to VGG16 for a fair comparison.
- **Training:** Adam optimizer, binary cross-entropy loss, early stopping, learning-rate reduction on plateau, model checkpointing.
- **Explainability:** Grad-CAM heatmaps to verify the model focuses on lung regions.

## Results

Evaluated on the independent 624-image test set (Pneumonia = positive class):

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| Custom CNN | 0.622 | 0.953 | 0.415 | 0.966 | 0.579 | 0.882 |
| **VGG16** | **0.925** | 0.905 | **0.982** | 0.829 | **0.942** | **0.974** |
| ResNet50 | 0.800 | 0.932 | 0.733 | 0.910 | 0.821 | 0.922 |

**VGG16 was the best model**, correctly identifying 383 of 390 pneumonia cases (only 7 false negatives). The custom CNN performed poorest, missing 228 of 390 pneumonia cases.

### Efficiency

| Model | Total Params | Trainable | Size (MB) | Inference (ms) |
|---|---|---|---|---|
| Custom CNN | 522,433 | 521,473 | 1.99 | 80.55 ± 3.41 |
| VGG16 | 14,879,041 | 7,243,777 | 56.76 | 86.62 ± 5.59 |
| ResNet50 | 24,145,281 | 5,023,233 | 92.11 | 89.53 ± 3.89 |

The smallest model (custom CNN) was the least accurate; the largest (ResNet50) was not the best. VGG16 offered the best accuracy-to-resources trade-off.

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and arrange it as:

```
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test/
```

Then open the notebook in `notebooks/` and run the cells, or run it directly on Kaggle.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── pneumonia_detection.ipynb
├── figures/
│   └── (confusion matrices, ROC curves, training curves, Grad-CAM, architecture)
├── report/
│   └── final_report.pdf
└── .gitignore
```

## Limitations

- Single paediatric population (ages 1–5) from one medical centre — limits generalisability.
- Very small validation split (16 images) → noisy validation signals.
- No external validation on independent datasets; single train/val/test split.
- Grad-CAM explanations were not reviewed by a clinical domain expert.

## Future Work

External validation on independent datasets, additional backbones (DenseNet, EfficientNet), threshold tuning and hyperparameter search to improve custom CNN sensitivity, SHAP attribution to complement Grad-CAM, and extension to bacterial vs. viral pneumonia classification.

## Citation

If referencing this work, please cite the original dataset:

> Kermany, D. S., et al. (2018). "Identifying medical diagnoses and treatable diseases by image-based deep learning." *Cell*, 172(5), 1122–1131.

## Disclaimer

This system is intended as a **decision-support tool with human supervision**, not as a standalone diagnostic tool. It should not be used for clinical decision-making without expert oversight.
