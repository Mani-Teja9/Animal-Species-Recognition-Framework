# 🫁 Chest X-Ray Pneumonia Detection System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**Student:** Mani Teja Padala | **ID:** 69767331
**Course:** Machine Learning — Phase 3 (Final Report & Presentation)
**University:** University of Europe for Applied Sciences, Potsdam, Germany

🔗 **Live Kaggle Notebook:** https://www.kaggle.com/code/padalamaniteja/notebookb89126fae2
🔗 **GitHub Repository:** https://github.com/Mani-Teja9/Chest-XRay-Pneumonia-Detection

---

## 📌 Project Overview

This project implements a CNN-based binary image classification system to detect **Pneumonia** from paediatric chest X-ray images. The study is **comparative and explainable**: three models are trained under identical, imbalance-aware conditions, with an emphasis on **sensitivity** (the clinically critical metric, since a missed pneumonia case is the most dangerous error) and on **explainability** via Grad-CAM.

1. **Custom CNN** — Built from scratch (4 Conv blocks)
2. **VGG16** — Transfer learning with two-stage fine-tuning
3. **ResNet50** — Transfer learning with two-stage fine-tuning

---

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle
🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

> Note: The dataset is **imbalanced** (~73% Pneumonia / ~27% Normal), handled with balanced class weights (Normal ≈ 1.945, Pneumonia ≈ 0.673) and augmentation. The validation split is intentionally tiny (16 images), so validation metrics during training are noisy. The **test set (624 images)** is the reliable basis for the reported results.

---

## ❓ Research Questions

- **RQ1:** Can a custom CNN trained from scratch achieve clinically acceptable accuracy (>90%) on pneumonia detection?
- **RQ2:** How do transfer-learning models (VGG16, ResNet50) compare to a custom CNN on accuracy, sensitivity, specificity, and AUC-ROC?
- **RQ3:** What is the impact of data augmentation and class balancing on imbalanced medical data?
- **RQ4:** Which X-ray regions are most important for classification? (Grad-CAM)
- **RQ5:** What is the trade-off between predictive performance and computational efficiency (model size, inference time)?

---

## 🧠 Models

### Custom CNN Architecture
```
Input (224×224×3)
→ Conv2D(32) + BatchNorm + MaxPool
→ Conv2D(64) + BatchNorm + MaxPool
→ Conv2D(128) + BatchNorm + MaxPool
→ Conv2D(256) + BatchNorm + MaxPool
→ GlobalAveragePooling2D
→ Dense(512, ReLU) + Dropout(0.5)
→ Dense(1, Sigmoid)
```

### Transfer Learning
Both pretrained models use a two-stage **freeze-then-fine-tune** strategy: the ImageNet base is first frozen while a new classification head is trained, then the later layers are unfrozen and fine-tuned at a lower learning rate.

| Model | Base | Strategy | Head LR | Fine-tune LR |
|-------|------|----------|---------|--------------|
| VGG16 | ImageNet | Freeze → fine-tune later conv layers | 1e-4 | lower (e.g. 1e-5) |
| ResNet50 | ImageNet | Freeze → fine-tune later conv layers | 1e-4 | lower (e.g. 1e-5) |

---

## 📈 Results (Independent 624-image Test Set)

Pneumonia = positive class. **VGG16 was the best and most clinically relevant model**, missing only **7 of 390** pneumonia cases.

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 | AUC-ROC |
|-------|----------|-----------|-------------|-------------|------|---------|
| Custom CNN | 0.622 | 0.953 | 0.415 | 0.966 | 0.579 | 0.882 |
| **VGG16** | **0.925** | 0.905 | **0.982** | 0.829 | **0.942** | **0.974** |
| ResNet50 | 0.800 | 0.932 | 0.733 | 0.910 | 0.821 | 0.922 |

> The custom CNN's high precision/specificity is **misleading**: it under-detected pneumonia, missing 228 of 390 cases (41.5% sensitivity) — the most dangerous failure mode for a screening tool. VGG16 missed only 7.

### ⚙️ Computational Efficiency

| Model | Total Params | Trainable | Size (MB) | Inference (ms) |
|-------|--------------|-----------|-----------|----------------|
| Custom CNN | 522,433 | 521,473 | 1.99 | 80.55 ± 3.41 |
| VGG16 | 14,879,041 | 7,243,777 | 56.76 | 86.62 ± 5.59 |
| ResNet50 | 24,145,281 | 5,023,233 | 92.11 | 89.53 ± 3.89 |

The smallest model (custom CNN) was the least accurate; the largest (ResNet50) was not the best. **VGG16 offered the best accuracy-to-resources trade-off.**

---

## 📋 Evaluation Metrics

- Accuracy, Precision, Recall (Sensitivity), Specificity
- F1-Score, AUC-ROC
- Confusion Matrix (per model)
- Grad-CAM Visualisations
- Efficiency: parameters, model size, inference time

---

## 📁 Repository Structure

```
📦 Chest-XRay-Pneumonia-Detection/
 ┣ 📓 chest_xray_pneumonia_detection.ipynb   ← Main Kaggle notebook
 ┣ 📄 ML_Phase2_Proposal_ManiTeja.docx       ← Project proposal
 ┣ 📄 Final_Report.pdf                        ← Phase 3 final report
 ┗ 📄 README.md                              ← This file
```

> **About figures:** All figures (class distribution, sample X-rays, augmented samples,
> training curves, confusion matrices, ROC comparison, Grad-CAM) are
> **generated automatically when the notebook runs**. You can view them directly in the
> Kaggle notebook output, or in the final report PDF. If you'd like them stored in this
> repository as well, download them from the Kaggle notebook's **Output** panel and add
> them to a `figures/` folder.

---

## 🚀 How to Run

### On Kaggle (Recommended)
1. Open the notebook: https://www.kaggle.com/code/padalamaniteja/notebookb89126fae2
   (or create a **New Notebook** and import `chest_xray_pneumonia_detection.ipynb`)
2. Attach the dataset: **+ Add Input** → search `chest-xray-pneumonia` → add the one by **paultimothymooney**
3. **Settings → Accelerator → GPU T4 ×2**
4. **Settings → Internet → On** (required — VGG16/ResNet50 download their ImageNet weights)
5. Click **Run All**

> ⚠️ On Kaggle, **do NOT \`pip install\` TensorFlow / Keras / scikit-learn / etc.** — they are
> already pre-installed, and reinstalling them can break the numpy/scipy environment.
> The notebook also **auto-detects** the dataset folder paths, so no manual path editing is needed.

### On Local Machine
```bash
# Clone repository
git clone https://github.com/Mani-Teja9/Chest-XRay-Pneumonia-Detection
cd Chest-XRay-Pneumonia-Detection

# Install dependencies
pip install tensorflow keras scikit-learn matplotlib seaborn opencv-python

# Download dataset from Kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/

# The notebook auto-detects the train/val/test folders.
jupyter notebook chest_xray_pneumonia_detection.ipynb
```

---

## 🛠️ Dependencies

```
tensorflow >= 2.10
keras >= 2.10
scikit-learn >= 1.0
matplotlib >= 3.5
seaborn >= 0.12
opencv-python >= 4.6
numpy >= 1.23
pandas >= 1.5
```

---

## ⚠️ Limitations & Future Work

- Single paediatric population (ages 1–5) from one medical centre — limits generalisability.
- Very small validation split (16 images) → noisy validation signals.
- No external validation on independent datasets; single train/val/test split.
- **Future work:** external validation, additional backbones (DenseNet, EfficientNet), threshold/hyperparameter tuning to raise custom-CNN sensitivity, SHAP attribution to complement Grad-CAM, and bacterial vs. viral pneumonia classification.

> **Disclaimer:** This system is intended as a **decision-support tool with human supervision**, not a standalone diagnostic tool.

---

## 📚 References

1. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122–1131. https://doi.org/10.1016/j.cell.2018.02.010
2. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR 2015*. https://arxiv.org/abs/1409.1556
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. https://arxiv.org/abs/1512.03385
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*. https://arxiv.org/abs/1610.02391

---

## 📄 License

Academic project — University of Europe for Applied Sciences, 2026.
