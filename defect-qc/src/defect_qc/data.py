import kagglehub
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Callable
import torch
from torchvision.transforms import v2
from PIL import Image


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
    

def get_data_dir():
    data_dir = Path(kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product"))
    data_dir = data_dir / "casting_data" / "casting_data"
    return data_dir


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


def build_loader(cfg, df: pd.DataFrame, train: bool = False):
    # create transformers
    transform = build_transforms(cfg, train=train)
    ds = CastingQCDataset(cfg, df, transform)
    dl = DataLoader(ds, 
                    cfg.data.batch_size, 
                    shuffle=train, 
                    num_workers=cfg.data.num_workers,
                    # pin_memory=True, not supported on MPS
                    persistent_workers=cfg.data.num_workers > 0) # it can crash if num_workers = 0
    return dl


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

 