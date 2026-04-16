from pathlib import Path
import copy
import numpy as np
import torch
import hydra
from hydra.utils import instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from defect_qc.data import get_data_dir, create_manifest, build_loader, build_dataframes
from defect_qc.eval import evaluate_binary_classification, save_confusion_matrix


def log_metrics(metrics: dict, prefix: str, step: int):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"{prefix}/{k}", float(v), step=step)


def log_model(cfg, model, in_channels: int):
    input_shape = (1, in_channels, cfg.data.img_size, cfg.data.img_size)
    input_example = torch.zeros(input_shape, dtype=torch.float32)
    model_for_logging = copy.deepcopy(model).to("cpu").eval()
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype(np.float32), input_shape)])
    )
    mlflow.pytorch.log_model(
        model_for_logging,
        name="model",
        input_example=input_example,
        signature=signature,
        serialization_format="pt2",
    )


def save_demo_checkpoint(cfg, model, in_channels: int) -> Path:
    checkpoint_path = Path(to_absolute_path(cfg.demo.checkpoint_path))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "data": {
            "img_mode": cfg.data.img_mode,
            "img_size": int(cfg.data.img_size),
            "in_channels": int(in_channels),
            "class_names": list(cfg.demo.class_names),
            "label_mapping": {"ok": 0, "defect": 1},
        },
        "evaluation": {
            "threshold": float(cfg.evaluation.threshold),
        },
        "run": OmegaConf.to_container(cfg.run, resolve=True),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def train_model(cfg, model, dl_train, dl_val, criterion, optimizer, scheduler, device):
    def _train_one_epoch():
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = yb.numel()
            loss_sum += loss.item() * batch_size
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == yb).sum().item() # take float from tensor
            total += batch_size

        train_out = {"loss": loss_sum / total, "acc": correct / total}

        return train_out

    for epoch in range(cfg.num_epochs):
        train_out = _train_one_epoch()
        val_out = evaluate_binary_classification(model, dl_val, criterion, device, cfg.evaluation.threshold)

        log_metrics(train_out, "train", epoch)
        log_metrics(val_out, "val", epoch)
        
        if scheduler is not None:
            scheduler.step(val_out["loss"])

        print(
                f"| Epoch {epoch + 1:>3}/{cfg.num_epochs:<3} | "
                f"Train Loss: {train_out['loss']:7.4f} | "
                f"Val Loss: {val_out['loss']:7.4f} | "
                f"Train Acc: {train_out['acc']:7.4f} | "
                f"Val Acc: {val_out['acc']:7.4f} |"
            )

def test_model(cfg, model, dl_test, criterion, device):

    test_out = evaluate_binary_classification(model, dl_test, criterion, device, cfg.evaluation.threshold, compute_cm=True)
 
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    cm_path = run_dir / "confusion_matrix_test.png"
    save_confusion_matrix(test_out["cm"], cm_path, class_names=("ok", "defect"))
    mlflow.log_artifact(str(cm_path), artifact_path="figures")

    log_metrics(test_out, "test", step=0)
    
    print(
        f"| TEST | "
        f"Acc: {test_out['acc']:7.4f} | "
        f"F1: {test_out['f1']:7.4f} | "
        f"Precision: {test_out['precision']:7.4f} | "
        f"Recall: {test_out['recall']:7.4f} |"

    )

    print(test_out["cm"])



@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.run.name):
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), artifact_file="config.yaml")

        df = create_manifest(get_data_dir())
        df_train, df_val, df_test = build_dataframes(cfg, df)
        
        dl_train = build_loader(cfg, df_train, train=True)
        dl_val = build_loader(cfg, df_val)
        dl_test = build_loader(cfg, df_test)

        # init model with hydra
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        in_channels = 3 if cfg.data.img_mode == "RGB" else 1
        cfg.model.in_channels = in_channels
        class_weights = torch.tensor(cfg.data.class_weights, dtype=torch.float32).to(device)

        model = instantiate(cfg.model, in_channels=in_channels).to(device)
        criterion = instantiate(cfg.criterion, weight=class_weights)
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

        train_model(cfg, model, dl_train, dl_val, criterion, optimizer, scheduler, device)
        test_model(cfg, model, dl_test, criterion, device)

        demo_checkpoint_path = save_demo_checkpoint(cfg, model, in_channels)
        print(f"Saved demo checkpoint: {demo_checkpoint_path}")

        # Log trained model to MLflow
        mlflow.log_artifact(str(demo_checkpoint_path), artifact_path="checkpoints")
        log_model(cfg, model, in_channels)


if __name__ == "__main__":
    main()
