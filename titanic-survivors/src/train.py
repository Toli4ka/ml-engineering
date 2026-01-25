from data import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import mlflow


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

def prepare_dataset(cfg):
    ds = load_dataset(DATA_DIR)
    target = cfg.data.target
    X = ds.drop(columns=[target])
    y = ds[target].astype(int) # convert bool to int
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=cfg.data.test_size, 
                                                      random_state=cfg.data.random_state, 
                                                      stratify=y, 
                                                      shuffle=cfg.data.shuffle)
    return X_train, X_val, y_train, y_val

def proproccessor_init(cfg, features):
    # Convert OmegaConf lists to Python lists
    drop_cols = list(cfg.features.drop_cols)
    numeric_cols = list(cfg.features.numeric_cols)
    categorical_cols = list(cfg.features.categorical_cols)
    
    # sanity checks
    all_cols = set(features)
    chosen = set(drop_cols) | set(numeric_cols) | set(categorical_cols)

    assert chosen <= all_cols, "One of your lists contains a column that isn't in X_train"
    expected_length = len(drop_cols) + len(numeric_cols) + len(categorical_cols)
    assert len(chosen) == expected_length, "A column appears in multiple lists"
    # pipeline init
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg.preprocess.numeric_imputer, 
                                  add_indicator=cfg.preprocess.numeric_add_indicator))
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg.preprocess.categorical_imputer)),
        ("onehot", OneHotEncoder(handle_unknown=cfg.preprocess.onehot_handle_unknown)) 
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",  # everything else gets dropped
    )
    return preprocessor

def evaluate_model(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm # numpy array (not for mlflow metrics)
    }


def save_confusion_matrix_plot(cm, path, labels=("Died", "Survived"), title="Confusion Matrix"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def validate(cfg, pipe, X_val, y_val):
    # y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= cfg.eval.threshold).astype(int)
    return evaluate_model(y_val, y_pred, y_proba)

def prefix_metrics(metrics: dict, prefix: str) -> dict:
    out = {}
    for k, v in metrics.items():
        # keep only numeric metrics for mlflow
        if isinstance(v, (int, float)):
            out[f"{prefix}_{k}"] = float(v)
    return out

def train(cfg, X_train, y_train):
    
    preprocessor = proproccessor_init(cfg, features=X_train.columns)
    model = RandomForestClassifier(n_estimators=cfg.model.params.n_estimators,
                                 class_weight=cfg.model.params.class_weight,
                                 random_state=cfg.model.params.random_state)
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)])
    
    # fit data and train model
    pipeline.fit(X_train, y_train)

    # evaluate on train set
    y_pred = pipeline.predict(X_train)
    y_proba = pipeline.predict_proba(X_train)[:, 1]

    metrics = evaluate_model(y_train, y_pred, y_proba)

    return pipeline, metrics

def log_run(pipeline, train_metrics, val_metrics):
        # train confusion matrix artifact
        save_confusion_matrix_plot(
            train_metrics["confusion_matrix"],
            "train_confusion_matrix.png",
            title="Train Confusion Matrix"
        )
        # val confusion matrix artifact
        save_confusion_matrix_plot(
            val_metrics["confusion_matrix"],
            "val_confusion_matrix.png",
            title="Validation Confusion Matrix"
        )
        mlflow.log_artifact("train_confusion_matrix.png")
        mlflow.log_artifact("val_confusion_matrix.png")
        mlflow.log_artifact("resolved_config.yaml")

        mlflow.log_metrics(prefix_metrics(train_metrics, "train"))
        mlflow.log_metrics(prefix_metrics(val_metrics, "val"))
        mlflow.sklearn.log_model(pipeline, "model")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    tracking_dir = Path(hydra.utils.get_original_cwd()) / "mlruns"
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.run.name):
        OmegaConf.save(cfg, "resolved_config.yaml", resolve=True)
        X_train, X_val, y_train, y_val = prepare_dataset(cfg)
        pipeline, train_metrics = train(cfg, X_train, y_train)
        val_metrics = validate(cfg, pipeline, X_val, y_val)

        log_run(pipeline, train_metrics, val_metrics)


if __name__ == "__main__":
    main()
