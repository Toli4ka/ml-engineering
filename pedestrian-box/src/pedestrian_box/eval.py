from ultralytics import YOLO
from pedestrian_box.utils import get_data_config_path, get_model_path

DATA_CONFIG_PATH = get_data_config_path()
MODEL_PATH = get_model_path()

def main():
    model = YOLO(MODEL_PATH)
    metrics = model.val(data=DATA_CONFIG_PATH, imgsz=640, device="mps")
    print(metrics)

if __name__ == "__main__":
    main()