# 🫁 Chest X-Ray Pneumonia Detection System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**Student:** Mani Teja Padala | **ID:** 69767331
**Course:** Machine Learning — Phase 2
**University:** University of Europe for Applied Sciences, Potsdam, Germany

🔗 **Live Kaggle Notebook:** https://www.kaggle.com/code/padalamaniteja/notebookb89126fae2
🔗 **GitHub Repository:** https://github.com/Mani-Teja9/Chest-XRay-Pneumonia-Detection

---

## 📌 Project Overview

This project implements a CNN-based binary image classification system to detect **Pneumonia** from chest X-ray images. Three models are trained and compared:

1. **Custom CNN** — Built from scratch (4 Conv blocks)
2. **VGG16** — Transfer learning with fine-tuning
3. **ResNet50** — Transfer learning with fine-tuning

---

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle
🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,863** |

> Note: This dataset's validation split is intentionally tiny (16 images), so validation metrics during training are noisy. The **test set (624 images)** is the reliable basis for the reported results.

---

## ❓ Research Questions

- **RQ1:** Can a custom CNN achieve >90% accuracy on pneumonia detection?
- **RQ2:** How does transfer learning compare to a custom CNN?
- **RQ3:** What is the impact of data augmentation on imbalanced medical data?
- **RQ4:** Which X-ray regions are most important for classification? (Grad-CAM)

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
| Model | Base | Fine-tuned Layers | LR |
|-------|------|-------------------|----|
| VGG16 | ImageNet | Last 4 layers | 1e-5 |
| ResNet50 | ImageNet | Last 10 layers | 1e-5 |

---

## 📈 Target Results (Reference)

These are the expected ranges used for comparison. **Actual measured results are produced when the notebook runs** and are shown in the model comparison table and ROC plots inside the Kaggle notebook output.

| Model | Accuracy | Sensitivity | AUC-ROC |
|-------|----------|-------------|---------|
| Custom CNN | ~90% | ~92% | ~0.95 |
| VGG16 | ~93% | ~95% | ~0.97 |
| ResNet50 | ~95% | ~96% | ~0.98 |

---

## 📋 Evaluation Metrics

- Accuracy, Precision, Recall (Sensitivity), Specificity
- F1-Score, AUC-ROC
- Confusion Matrix
- Grad-CAM Visualisations

---

## 📁 Repository Structure

```
📦 Chest-XRay-Pneumonia-Detection/
 ┣ 📓 chest_xray_pneumonia_detection.ipynb   ← Main Kaggle notebook
 ┣ 📄 ML_Phase2_Proposal_ManiTeja.docx       ← Project proposal
 ┗ 📄 README.md                              ← This file
```

> **About figures:** All figures (class distribution, sample X-rays, augmented samples,
> training curves, confusion matrices, ROC comparison, Grad-CAM) and `model_comparison.csv`
> are **generated automatically when the notebook runs**. You can view them directly in the
> Kaggle notebook output. If you'd like them stored in this repository as well, download them
> from the Kaggle notebook's **Output** panel and add them to a `figures/` folder.

---

## 🚀 How to Run

### On Kaggle (Recommended)
1. Open the notebook: https://www.kaggle.com/code/padalamaniteja/notebookb89126fae2
   (or create a **New Notebook** and import `chest_xray_pneumonia_detection.ipynb`)
2. Attach the dataset: **+ Add Input** → search `chest-xray-pneumonia` → add the one by **paultimothymooney**
3. **Settings → Accelerator → GPU T4 ×2**
4. **Settings → Internet → On** (required — VGG16/ResNet50 download their ImageNet weights)
5. Click **Run All**

> ⚠️ On Kaggle, **do NOT `pip install` TensorFlow / Keras / scikit-learn / etc.** — they are
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
# If running fully offline, point find_split(search_root=...) to your local data path
# (e.g. './data') in the dataset-loading cell.
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

## 📚 References

1. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122–1131. https://doi.org/10.1016/j.cell.2018.02.010
2. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR 2015*. https://arxiv.org/abs/1409.1556
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. https://arxiv.org/abs/1512.03385
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV 2017*. https://arxiv.org/abs/1610.02391

---

## 📄 License

Academic project — University of Europe for Applied Sciences, 2026.
