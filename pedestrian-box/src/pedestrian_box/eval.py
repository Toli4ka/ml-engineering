from ultralytics import YOLO
from pathlib import Path

DATA_CONFIG_PATH = Path("/Users/anatolii/Projects/ml-engineering/pedestrian-box/data/data.yml")

def main():
    model = YOLO("yolo26n.pt")
    metrics = model.val(data=DATA_CONFIG_PATH, imgsz=640, device="mps")
    print(metrics)

if __name__ == "__main__":
    main()