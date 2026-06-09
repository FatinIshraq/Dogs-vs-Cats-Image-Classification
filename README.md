# Dogs vs Cats — Image Classification with EfficientNetV2S Fine-Tuning

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Dogs%20vs%20Cats-lightgrey)
![Accuracy](https://img.shields.io/badge/Accuracy->99%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

Binary image classifier distinguishing dogs from cats by fine-tuning EfficientNetV2S on ImageNet pre-trained weights. Achieves >99% training and validation accuracy with full performance analysis including confusion matrix, precision, recall, and F1 score.

---

## Results

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | ~99.5% | ~99.1% |
| Final Loss | ~0.025 | ~0.015 |

### Training Curves
![Training Curves](results/training_curves.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

---

## Approach

Rather than training a CNN from scratch, this project leverages **transfer learning** — using EfficientNetV2S pre-trained on ImageNet as a feature extractor, then fine-tuning all layers end-to-end on the target dataset. This is the standard industry approach for small datasets where training from scratch would overfit.

```
ImageNet Pre-trained EfficientNetV2S (all layers trainable)
         ↓
    top_activation layer output
         ↓
    Flatten
         ↓
    Dense(1024, ReLU)
         ↓
    Dropout(0.2)
         ↓
    Dense(1, Sigmoid)   →   Cat (0) / Dog (1)
```

---

## Dataset

- **Source:** [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) (2,000 image subset)
- **Split:** 1,000 train / 1,000 validation (500 cats + 500 dogs each)
- **Input size:** 260 × 260 × 3

---

## Data Augmentation

Applied to training set only to reduce overfitting:

| Augmentation | Value |
|-------------|-------|
| Rescale | 1/255 |
| Rotation | ±30° |
| Width / Height shift | ±20% |
| Shear | 0.2 |
| Zoom | 0.2 |
| Horizontal flip | Yes |

---

## Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Base model | EfficientNetV2S (ImageNet weights) |
| All layers trainable | Yes (full fine-tuning) |
| Optimizer | RMSprop (lr=1e-4) |
| Loss | Binary Cross-Entropy |
| Batch size | 20 |
| Epochs | 10 |
| Steps per epoch | 100 |
| Decision threshold | 0.6 |

---

## Performance Analysis

Evaluated on both train and validation sets using:
- Confusion Matrix
- Precision / Recall / F1 Score (`sklearn.metrics`)


---

## Project Structure

```
Dogs-vs-Cats-Image-Classification/
├── notebook/
│   └── dogs_vs_cats_classification.ipynb   # Full pipeline: EDA → fine-tuning → evaluation
├── results/
│   ├── training_curves.png                  # Training vs validation accuracy
│   └── confusion_matrix.png               # Confusion matrix on validation set
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/FatinIshraq/Dogs-vs-Cats-Image-Classification
cd Dogs-vs-Cats-Image-Classification
pip install -r requirements.txt
jupyter notebook notebook/dogs_vs_cats_classification.ipynb
```

The notebook automatically downloads the dataset on first run.

---

## Key Concepts Demonstrated

- Transfer learning with ImageNet pre-trained weights
- Full fine-tuning vs feature extraction tradeoffs
- `ImageDataGenerator` with augmentation for small datasets
- Binary classification with sigmoid output and custom threshold (0.6)
- End-to-end performance evaluation: confusion matrix, precision, recall, F1

---

## Roadmap

- [x] EfficientNetV2S fine-tuning (>99% accuracy)
- [x] Confusion matrix & classification report
- [ ] Grad-CAM visualization — which image regions drive predictions
- [ ] Comparison: EfficientNetV2S vs ResNet50 vs VGG16
- [ ] Full 25,000-image Kaggle dataset

---

## Author

**Fatin Ishraq** 

[![GitHub](https://img.shields.io/badge/GitHub-FatinIshraq-black?logo=github)](https://github.com/FatinIshraq)
