# 🐒 Explainable Animal Species Recognition (10 Monkey Species)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**Student:** Mani Teja Padala | **ID:** 69767331
**Course:** Machine Learning
**University:** University of Europe for Applied Sciences, Potsdam, Germany

---

## 📌 Project Overview

This project implements a CNN-based **ten-class image classification** system to recognise
**monkey species** from photographs, with **Grad-CAM** explainability. Three models are
trained under identical, augmentation-based conditions and compared fairly:

1. **Custom CNN** — built from scratch (4 Conv blocks)
2. **VGG16** — transfer learning with fine-tuning
3. **ResNet50** — transfer learning with fine-tuning

---

## 📊 Dataset

**10 Monkey Species** — Kaggle (slothkong)
🔗 https://www.kaggle.com/datasets/slothkong/10-monkey-species

| Split | Images | Notes |
|-------|--------|-------|
| Train | ~1,098 | after 15% internal val split |
| Internal Validation | ~194 | 15% of training folder |
| Test (held-out) | 272 | dataset's own validation folder |
| **Total** | **~1,370** | 10 classes (n0–n9), JPEG, 224×224 RGB |

The ten classes are approximately balanced.

---

## ❓ Research Questions

- **RQ1:** Can a custom CNN trained from scratch achieve high accuracy on ten-class monkey recognition?
- **RQ2:** How do transfer-learning models (VGG16, ResNet50) compare to the custom CNN?
- **RQ3:** What is the effect of data augmentation on a small, multi-class dataset, and how do models behave per class?
- **RQ4:** Which image regions are most important for classification? (Grad-CAM)
- **RQ5:** What is the trade-off between predictive performance and computational efficiency?

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
→ Dense(10, Softmax)
```

### Transfer Learning
| Model | Base | Fine-tuned Layers | Fine-tuning LR |
|-------|------|-------------------|----------------|
| VGG16 | ImageNet | Last 4 conv layers | 1e-5 |
| ResNet50 | ImageNet | Last 10 layers | 1e-5 |

Both use a shared classification head and a two-stage **freeze-then-fine-tune** strategy.

---

## 📈 Results (held-out test set)

| Model | Accuracy | Macro F1 | AUC |
|-------|----------|----------|-----|
| Custom CNN | 0.110 | 0.058 | 0.539 |
| **VGG16** | **0.558** | **0.538** | **0.887** |
| ResNet50 | 0.164 | 0.087 | 0.600 |

**Conclusion:** Transfer learning, specifically **VGG16**, is markedly more effective than a
shallow custom network for fine-grained species recognition on a small dataset, and offers
the best accuracy-to-resources trade-off. Grad-CAM confirmed the best model attends to the
animal rather than the background.

---

## 📋 Evaluation Metrics

- Accuracy, Macro Precision, Recall, F1-Score
- AUC-ROC (one-vs-rest)
- Confusion Matrices (per model)
- Efficiency: parameter count, model size (MB), inference time
- Grad-CAM Visualisations

---

## 📁 Repository Structure

```
📦 Animal-Species-Recognition/
 ┣ 📓 notebooka2f4170983.ipynb               ← Main Kaggle notebook (full pipeline)
 ┣ 📄 README.md                              ← This file
 ┗ 📂 figures/                               ← Generated output figures
     ┣ sample_images.png
     ┣ custom_cnn_curves.png
     ┣ vgg16_transfer_learning_curves.png
     ┣ resnet50_transfer_learning_curves.png
     ┣ custom_cnn_confusion.png
     ┣ vgg16_confusion.png
     ┣ resnet50_confusion.png
     ┣ roc_comparison.png
     ┗ gradcam.png
```

---

## 🚀 How to Run

### On Kaggle (Recommended)
1. Open the notebook: https://www.kaggle.com/code/padalamaniteja/notebooka2f4170983
2. Add the dataset: https://www.kaggle.com/datasets/slothkong/10-monkey-species
3. Enable GPU: Settings → Accelerator → GPU
4. Click **Run All**

### On Local Machine
```bash
# Clone repository
git clone https://github.com/Mani-Teja9/Animal-Species-Recognition
cd Animal-Species-Recognition

# Install dependencies
pip install tensorflow keras scikit-learn matplotlib seaborn opencv-python

# Download dataset from Kaggle
kaggle datasets download -d slothkong/10-monkey-species
unzip 10-monkey-species.zip -d data/

# Update the data path in the notebook, then run
jupyter notebook notebooka2f4170983.ipynb
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

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
2. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR 2015.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV 2017.
5. Pillai, R., et al. (2023). A Deep Learning Approach for Detection and Classification of Ten Species of Monkeys. ICSSES 2023.
6. slothkong (2018). 10 Monkey Species. Kaggle.

---

## 📄 License

Academic project — University of Europe for Applied Sciences, 2026.
