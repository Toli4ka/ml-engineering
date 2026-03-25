from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO


def get_default_model_path():
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "yolo26n.pt"

    if not model_path.exists():
        raise FileNotFoundError("No YOLO weights file found in the expected project locations.")
    
    return model_path

    


@lru_cache(maxsize=4)
def load_yolo_model(model_path=None):
    resolved_model_path = Path(model_path) if model_path else get_default_model_path()
    return YOLO(str(resolved_model_path))


def predict(model, image_path, conf=0.25, iou=0.45):
    return model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)[0]
