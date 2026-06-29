# Animal Species Recognition Framework

CNN-based image classification project for the 10 Monkey Species Kaggle dataset.

## Project Summary

This repository compares three CNN approaches:

- Custom CNN trained from scratch
- VGG16 transfer learning with fine-tuning
- ResNet50 transfer learning with fine-tuning

The best model in the Kaggle run was VGG16 with 55.51% test accuracy, 0.5344 macro F1-score, and 0.8864 AUC.

## Dataset

Kaggle dataset: https://www.kaggle.com/datasets/slothkong/10-monkey-species

The implementation uses 1,098 training images and 272 independent test images across 10 monkey species classes.

## Contents

- `notebooks/animal_species_recognition_framework.ipynb` - completed Kaggle notebook
- `figures/` - exported training curves, confusion matrices, ROC curve, Grad-CAM, and methodology figure

## Kaggle Notebook

https://www.kaggle.com/code/padalamaniteja/notebooka2f4170983

## How to Reproduce

1. Open the Kaggle notebook.
2. Add the `10 Monkey Species` dataset as input.
3. Enable GPU acceleration.
4. Run all cells to reproduce model training, evaluation, figures, and tables.

