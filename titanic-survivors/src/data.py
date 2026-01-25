import pandas as pd
from pathlib import Path
import kagglehub


def load_dataset(data_dir: Path):
    download_from_kagglehub(data_dir)
    return pd.read_csv(data_dir / "Titanic-Dataset.csv")

def download_from_kagglehub(dir_to_save: Path):
        # Download to default kagglehub cache
    path = kagglehub.dataset_download("yasserh/titanic-dataset")
    # Copy to the data folder
    source_path = Path(path)
    for file in source_path.iterdir():
        if file.is_file():
            destination = dir_to_save / file.name
            print(destination)
            if not destination.exists():
                destination.write_bytes(file.read_bytes())
                print(f"Copied {file.name} to {dir_to_save}")
            else:
                print(f"File {file.name} already exists in {dir_to_save}")  