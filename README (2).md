# 🫁 Chest X-Ray Pneumonia Detection System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**Student:** Mani Teja Padala | **ID:** 69767331
**Course:** Machine Learning — Phase 2
**University:** University of Europe for Applied Sciences, Potsdam, Germany

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

## 📈 Expected Results

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
 ┣ 📄 README.md                              ← This file
 ┣ 📂 figures/                               ← Generated output figures
 │   ┣ class_distribution.png
 │   ┣ sample_xray_images.png
 │   ┣ augmented_samples.png
 │   ┣ custom_cnn_curves.png
 │   ┣ vgg16_transfer_learning_curves.png
 │   ┣ resnet50_transfer_learning_curves.png
 │   ┣ custom_cnn_confusion_matrix.png
 │   ┣ vgg16_confusion_matrix.png
 │   ┣ resnet50_confusion_matrix.png
 │   ┣ roc_curves_comparison.png
 │   ┣ model_comparison_chart.png
 │   ┗ gradcam_visualisations.png
 ┗ 📄 model_comparison.csv
```

---

## 🚀 How to Run

### On Kaggle (Recommended)
1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click **New Notebook**
3. Upload `chest_xray_pneumonia_detection.ipynb`
4. Enable GPU accelerator: Settings → Accelerator → GPU
5. Click **Run All**

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

# Update BASE_DIR in notebook to: './data/chest_xray'
# Then run the notebook
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

1. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122–1131.
2. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks. ICLR 2015.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
4. Selvaraju, R. R., et al. (2017). Grad-CAM. ICCV 2017.

---

## 📄 License

Academic project — University of Europe for Applied Sciences, 2026.
