# Titanic Survivors: Production-Style ML Pipeline

This project was built as a hands-on exercise in designing a machine learning workflow that looks closer to production work than a one-off notebook. The dataset is the classic Titanic survival prediction task, but the main goal here was not the competition itself. The goal was to practice building a reproducible training pipeline with configuration management, experiment tracking, and a preprocessing flow that can be reused safely at training and inference time.

The project combines:

- `scikit-learn` pipelines and `ColumnTransformer` for reproducible preprocessing and model training
- `Hydra` for experiment configuration and parameter overrides
- `MLflow` for experiment tracking, metric logging, artifact logging, and model storage
- A simple project layout that separates data loading, training, and analysis

## Project Goal

The focus of this repository was to learn how to structure an ML project in a more professional way:

- keep experiments configurable instead of hard-coding parameters
- make preprocessing reproducible and versionable
- compare runs with tracked metrics instead of manual notes
- save artifacts that are useful for debugging and model review

## Pipeline Overview

The training pipeline in [`src/train.py`](/Users/anatolii/Projects/ml-engineering/titanic-survivors/src/train.py) follows a straightforward production-oriented structure:

1. Load the Titanic dataset.
2. Split data into train and validation sets with a fixed random seed and stratification.
3. Build a preprocessing pipeline:
   - numeric features: median imputation
   - categorical features: most frequent imputation + one-hot encoding
4. Train a `RandomForestClassifier`.
5. Evaluate both train and validation performance.
6. Log metrics, configuration, confusion matrices, and the trained model to MLflow.

The preprocessing is part of the sklearn pipeline, which means the same transformations used during training are bundled with the trained model. That is one of the main reasons this setup is much safer than doing preprocessing manually in notebooks.

## Configuration

Experiment settings are managed in [`configs/config.yaml`](/Users/anatolii/Projects/ml-engineering/titanic-survivors/configs/config.yaml). The current configuration includes:

- dataset split settings
- target column
- selected numeric, categorical, and dropped features
- preprocessing choices
- evaluation threshold
- model hyperparameters
- MLflow experiment name

This makes it easy to rerun experiments with explicit configuration changes instead of modifying code.

Example settings in the current run:

- model: `RandomForestClassifier`
- estimators: `300`
- class weight: `balanced`
- validation threshold: `0.4`
- dropped columns: `PassengerId`, `Name`, `Ticket`, `Cabin`, `Fare`
- numeric columns: `Age`, `SibSp`, `Parch`, `Pclass`
- categorical columns: `Sex`, `Embarked`

## Experiment Tracking

MLflow is used to log:

- train and validation metrics
- resolved Hydra configuration
- confusion matrix plots
- trained sklearn model artifact

Tracked artifacts can be found under the local MLflow outputs in [`mlruns`](/Users/anatolii/Projects/ml-engineering/titanic-survivors/mlruns) and experiment outputs in [`outputs`](/Users/anatolii/Projects/ml-engineering/titanic-survivors/outputs).

## Results

Two main experiment variants were tracked locally:

- `rf_baseline`
- `rf_reduced_threshold`

The stronger run was `rf_reduced_threshold`, which keeps the same Random Forest model but lowers the classification threshold to `0.4` during validation.

| Run | Val Accuracy | Val F1 | Val Precision | Val Recall | Val ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rf_baseline` | 0.8268 | 0.7669 | 0.7969 | 0.7391 | 0.8371 |
| `rf_reduced_threshold` | 0.8380 | 0.7883 | 0.7941 | 0.7826 | 0.8371 |

Takeaway:

- lowering the decision threshold improved validation recall and F1 score
- precision stayed nearly unchanged
- ROC AUC stayed the same because the underlying ranking quality of the model did not change, only the classification cutoff did

For this project, that was a useful reminder that model quality is not only about the estimator itself. Post-processing decisions such as threshold selection can materially change business-facing performance.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training locally with the default file-based MLflow behavior:

```bash
python src/train.py
```

Run with Hydra overrides:

```bash
python src/train.py eval.threshold=0.5 model.params.n_estimators=500 run.name=rf_threshold_05
```


## Repository Structure

```text
.
├── configs/
│   └── config.yaml
├── data/
├── mlruns/
├── outputs/
├── src/
│   ├── data.py
│   ├── train.py
│   ├── data_analysis.ipynb
│   └── model_analysis.ipynb
└── requirements.txt
```

## What This Project Demonstrates

- reproducible preprocessing with sklearn pipelines
- configurable experimentation with Hydra
- experiment tracking and artifact logging with MLflow
- separation between training logic, configuration, and analysis
- a workflow that is closer to real ML engineering practice than a notebook-only approach

## Future Improvements

The current version is a solid learning project, but there are several obvious next steps to make it more production-ready:

- add proper unit and integration tests for data loading, preprocessing, and training
- add cross-validation and a more systematic hyperparameter search
- add model registry usage and stage promotion in MLflow
- add data validation and schema checks before training
- package the training and inference code into a reusable module or CLI
- create an inference script or small API for serving predictions
- add feature importance or explainability reports
- add CI to run tests and validate configuration changes automatically
- track datasets and model versions more explicitly for stronger reproducibility

## Notes

- The dataset is downloaded via `kagglehub` in [`src/data.py`](/Users/anatolii/Projects/ml-engineering/titanic-survivors/src/data.py) and cached locally in the project `data` directory.
- The notebooks in [`src/data_analysis.ipynb`](/Users/anatolii/Projects/ml-engineering/titanic-survivors/src/data_analysis.ipynb) and [`src/model_analysis.ipynb`](/Users/anatolii/Projects/ml-engineering/titanic-survivors/src/model_analysis.ipynb) support exploratory analysis and run inspection.