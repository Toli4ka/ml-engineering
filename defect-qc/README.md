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
├── configs/                    # Hydra configuration files
├── src/
│   └── defect_qc/              # Python package
│       ├── __init__.py
│       ├── train.py            # Training entry point
│       ├── models.py           # CNN model definition
│       ├── eval.py             # Evaluation logic
│       └── data.py             # Data loading & transforms
│
├── tests/                      # Pytest test suite
├── pyproject.toml              # Project metadata & dependencies
├── .github/workflows/          # CI configuration
├── outputs/                    # Hydra run folders
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

## 🚀 Installation & Usage

### 1️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install project (editable mode)

```bash
pip install -e ".[dev]"
```

This installs the project as a package (`defect_qc`) and includes development dependencies (pytest, linting tools).

---

### 3️⃣ Run training

```bash
python -m defect_qc.train
```

Override parameters via Hydra:

```bash
python -m defect_qc.train data.img_size=128 num_epochs=20
```

---

## 🧪 Testing

Run unit tests:

```bash
pytest
```

---

## 🔄 Continuous Integration

GitHub Actions automatically runs on every push and pull request.

CI pipeline:

- Installs the package in a clean environment
- Runs the test suite
- Ensures reproducible installation from `pyproject.toml`

This guarantees that the project builds and tests successfully outside the local environment.

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
