from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_data_config_path() -> Path:
    return get_project_root() / "data" / "data.yml"

def get_model_path() -> Path:
    return get_project_root() / "models" / "yolo26n.pt"