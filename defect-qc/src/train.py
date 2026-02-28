from pathlib import Path
import kagglehub
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Callable
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix
import mlflow
import seaborn as sns
import numpy as np


DATA_DIR = Path(kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product"))
DATA_DIR = DATA_DIR / "casting_data" / "casting_data"

class CastingQCDataset(Dataset):
    def __init__(self, cfg, df: pd.DataFrame, transform: Callable):
        self.cfg = cfg
        self.transform = transform
        # Filter once and reset index so __getitem__ is stable
        self.df = df.reset_index(drop=True)
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = row[self.cfg.data.path_col]
        y = row[self.cfg.data.label_col]
        with Image.open(path) as img:
            img = img.convert(self.cfg.data.img_mode)
            x = self.transform(img)

        return x, torch.tensor(y, dtype=torch.long)
    

def create_manifest(data_path):
    data = []
    for split in ["train", "test"]:
        for label_folder in ["def_front", "ok_front"]:
            # create bool label (1: defect, 0: ok)
            label = 1 if label_folder == "def_front" else 0
            folder_path = data_path / split / label_folder
            for file_path in folder_path.glob("*.jpeg"):
                data.append({
                    "file_path": str(file_path),
                    "defect": label,
                    "split": split
                })  
    return pd.DataFrame(data)


def build_dataframes(cfg, df: pd.DataFrame):
    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()
    df_train, df_val = train_test_split(df_train, 
                                        test_size = cfg.data.val_size, 
                                        random_state = cfg.data.random_state, 
                                        stratify=df_train[cfg.data.label_col])
    df_val = df_val.copy()
    df_val["split"] = "val"
    return df_train, df_val, df_test


def build_transforms(cfg, train: bool):
    # check img mode
    if cfg.data.img_mode not in {"RGB", "L"}:
        raise ValueError(cfg.data.img_mode)
    # Normalizatoin (compute later dataset std and mean)
    if cfg.data.img_mode == "RGB":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean, std = (0.5,), (0.5,) # 1 channel for "L"

    ops = [v2.Resize((cfg.data.img_size, cfg.data.img_size))]
    # augment if train set:
    if train: 
        ops += [
            v2.RandomHorizontalFlip(p=cfg.data.augment.random_flip),
            v2.ColorJitter(brightness=cfg.data.augment.color_jitter_br, 
                           contrast=cfg.data.augment.color_jitter_ct)
        ]
    ops += [
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ]
    return v2.Compose(ops)

 
def build_loader(cfg, df: pd.DataFrame, train: bool = False):
    # create transformers
    transform = build_transforms(cfg, train=train)
    ds = CastingQCDataset(cfg, df, transform)
    dl = DataLoader(ds, 
                    cfg.data.batch_size, 
                    shuffle=train, 
                    num_workers=cfg.data.num_workers,
                    # pin_memory=True, not supported on MPS
                    persistent_workers=True)
    return dl


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
            prob = torch.softmax(logits, dim=-1)[:,1]
            preds = (prob >= cfg.evaluation.threshold).long()
            correct += (preds == yb).sum().item() # take float from tensor
            total += batch_size

        return loss_sum / total, correct / total

    def _evaluate():
        f1_metric = BinaryF1Score().to(device)
        precision_metric = BinaryPrecision().to(device)
        recall_metric = BinaryRecall().to(device)

        model.eval()
        with torch.no_grad():
            loss_sum, correct, total = 0.0, 0, 0
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)

                batch_size = yb.numel()
                loss_sum += loss.item() * batch_size
                prob = torch.softmax(logits, dim=-1)[:,1]
                preds = (prob >= cfg.evaluation.threshold).long()
                correct += (preds == yb).sum().item()
                total += batch_size

                # Update metrics
                f1_metric.update(preds, yb)
                precision_metric.update(preds, yb)
                recall_metric.update(preds, yb)
	
            f1 = f1_metric.compute().item()
            precision = precision_metric.compute().item()
            recall = recall_metric.compute().item()

            mlflow.log_metric("val/precision", precision, step=epoch)
            mlflow.log_metric("val/recall", recall, step=epoch)
            mlflow.log_metric("val/f1", f1, step=epoch)

        return loss_sum / total, correct / total

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = _train_one_epoch()
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = _evaluate()
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        mlflow.log_metric("train/loss", train_loss, step=epoch)
        mlflow.log_metric("train/acc", train_acc, step=epoch)
        mlflow.log_metric("val/loss", val_loss, step=epoch)
        mlflow.log_metric("val/acc", val_acc, step=epoch)
        
        if scheduler is not None:
            scheduler.step(val_loss)

        print(
                f"| Epoch {epoch + 1:>3}/{cfg.num_epochs:<3} | "
                f"Train Loss: {train_loss:7.4f} | "
                f"Val Loss: {val_loss:7.4f} | "
                f"Train Acc: {train_acc:7.4f} | "
                f"Val Acc: {val_acc:7.4f} |"
            )

def test_model(cfg, model, dl_test, criterion, device):
    f1_metric = BinaryF1Score().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    cm_metric = BinaryConfusionMatrix(threshold = cfg.evaluation.threshold).to(device)

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for xb, yb in dl_test:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)

            batch_size = yb.numel()
            prob = torch.softmax(logits, dim=-1)[:,1]
            preds = (prob >= cfg.evaluation.threshold).long()
            correct += (preds == yb).sum().item()
            total += batch_size

            # Update metrics
            f1_metric.update(preds, yb)
            precision_metric.update(preds, yb)
            recall_metric.update(preds, yb)
            cm_metric.update(prob, yb)

        test_acc = correct / total
        f1 = f1_metric.compute().item()
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        cm = cm_metric.compute().detach().cpu()
 
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        cm_path = run_dir / "confusion_matrix_test.png"
        save_confusion_matrix(cm, cm_path, class_names=("ok", "defect"))
        mlflow.log_artifact(str(cm_path), artifact_path="figures")

        mlflow.log_metric("test/acc", test_acc)
        mlflow.log_metric("test/f1", f1)
        mlflow.log_metric("test/precision", precision)
        mlflow.log_metric("test/recall", recall)
        
    print(
        f"| TEST | "
        f"Acc: {test_acc:7.4f} | "
        f"F1: {f1:7.4f} | "
        f"Precision: {precision:7.4f} | "
        f"Recall: {recall:7.4f} |"

    )

    print(cm)

def save_confusion_matrix(cm, path, class_names):
    fig, ax = plt.subplots(figsize=(5, 5))

    if isinstance(cm, torch.Tensor):
        cm_np = cm.detach().cpu().numpy()
    else:
        cm_np = np.asarray(cm)
    total = cm_np.sum()
    cm_percent = cm_np / cm_np.sum(axis=1, keepdims=True) * 100

    # Create annotation strings like "92.3%\n(426)"
    annot = np.empty_like(cm_np).astype(str)
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            annot[i, j] = f"{cm_percent[i, j]:.1f}%\n({cm_np[i, j]})"

    sns.heatmap(
        cm_np,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.run.name):
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), artifact_file="config.yaml")

        df = create_manifest(DATA_DIR)
        df_train, df_val, df_test = build_dataframes(cfg, df)
        
        dl_train = build_loader(cfg, df_train, train=True)
        dl_val = build_loader(cfg, df_val)
        dl_test = build_loader(cfg, df_test)

        # init model with hydra
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        in_channels = 3 if cfg.data.img_mode == "RGB" else 1
        cfg.model.in_channels = in_channels
        class_weights = torch.tensor([1.154, 0.882], dtype=torch.float32).to(device)

        model = instantiate(cfg.model, in_channels=in_channels).to(device)
        criterion = instantiate(cfg.criterion, weight=class_weights)
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

        train_model(cfg, model, dl_train, dl_val, criterion, optimizer, scheduler, device)
        test_model(cfg, model, dl_test, criterion, device)

if __name__ == "__main__":
    main()