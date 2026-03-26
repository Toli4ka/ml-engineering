from ultralytics import YOLO
from pedestrian_box.utils import get_data_config_path, get_model_path

DATA_CONFIG_PATH = get_data_config_path()
MODEL_PATH = get_model_path()


def run_inference(image_path, conf=0.25, iou=0.45):
    model = YOLO(MODEL_PATH)
    return model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)[0]


def extract_predictions(result):
    detections = []
    names = result.names if hasattr(result, "names") else {}

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

        detections.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "confidence": confidence,
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
            }
        )

    return detections