# 🏭 Defect Quality Control — Industrial Image Classification

End‑to‑end ML engineering project for binary image classification of casting manufacturing defects using PyTorch, Hydra, and MLflow.

This project demonstrates a production‑oriented workflow: reproducible experiments, configurable training, proper evaluation, and experiment tracking.

---

## 📌 Problem Statement

In casting manufacturing, visual defects must be detected reliably to avoid shipping faulty products. This project builds a CNN-based binary classifier that predicts:

- **0 → OK product**
- **1 → Defective product**

The focus is not only on modeling accuracy, but on building a clean, reproducible ML pipeline.

Dataset source:
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

---

## 🗂 Dataset

The dataset contains grayscale images of casting products stored as RGB images.

Structure:

```
casting_data/
    train/
        ok_front/
        def_front/
    test/
        ok_front/
        def_front/
```

The author-provided train/test split is used.

---

## ⚙️ Project Structure

```
defect-qc/
│
├── configs/                # Hydra configuration files
├── src/
│   ├── train.py            # Training + evaluation pipeline
│   ├── models.py           # CNN model definition
│   └── ...
│
├── outputs/                # Hydra run folders
├── mlflow/                 # Shared MLflow backend (optional)
└── README.md
```

Each Hydra run produces:

```
outputs/YYYY-MM-DD/HH-MM-SS/
    .hydra/
        config.yaml
        overrides.yaml
    confusion_matrix_test.png
```

---

## 🧠 Modeling Approach

### Data Pipeline
- Custom `Dataset` class
- Torch `DataLoader`
- Configurable image mode (`RGB` or `L`)
- Configurable image size
- Train-only augmentation:
  - Random horizontal flip
  - Color jitter
- Normalization

### Model
- Custom CNN baseline
- Adaptive average pooling
- Fully connected classifier head

### Loss
- `CrossEntropyLoss`
- Class weights to mitigate imbalance

### Optimizer
- AdamW

### Scheduler
- ReduceLROnPlateau

---

## 📊 Evaluation Metrics

Computed on validation and test sets:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix (counts + percentages)

Special focus: **Minimizing False Negatives** (missing defective products).

---

## 🔁 Reproducibility

- Hydra configuration system
- Automatic config snapshot per run
- Fixed random seeds
- MLflow experiment tracking
- Versioned artifacts (config, confusion matrix, metrics)

---

## 📈 MLflow Integration

The project logs:

- Training & validation metrics per epoch
- Final test metrics
- Resolved Hydra config
- Confusion matrix image

---

## 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python src/train.py
```

Override parameters via Hydra:

```bash
python src/train.py data.img_size=128 num_epochs=20
```

---

## 📌 Key Engineering Concepts Demonstrated

- Clean dataset abstraction
- Proper train/val/test split handling
- Avoiding data leakage
- Threshold-based evaluation
- TorchMetrics integration
- Centralized MLflow tracking server
- Artifact logging
- Config-driven experiments

---

## 📊 Results

Final model performance on the held‑out test set:

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.98   |
| Precision  | 0.99   |
| Recall     | 0.98   |
| F1 Score   | 0.99   |

Confusion Matrix:

<img src="outputs/2026-02-28/13-50-17/confusion_matrix_test.png" width="300">

### Interpretation

- The model achieves high precision, indicating a low number of false defect alarms.
- Recall reflects how many defective items are correctly detected.
- In industrial settings, threshold tuning may prioritize recall to further reduce false negatives.

---

## 🧩 Future Improvements

- ROC-AUC metric
- Threshold tuning via validation set
- Stronger backbone (ResNet / EfficientNet)
- Model export (TorchScript / ONNX)
- Inference CLI (`predict.py`)
- Dockerized MLflow server

---

## 🎯 Goal of This Project

This repository is part of a structured ML Engineering learning roadmap focused on:

- Shipping small but complete ML systems
- Practicing production-like workflows
- Developing experiment discipline
