from pathlib import Path

from pedestrian_box.model import load_yolo_model, predict


def run_inference(image_path, model_path=None, conf=0.25, iou=0.45):
    model = load_yolo_model(str(model_path) if model_path else None)
    return predict(model=model, image_path=Path(image_path), conf=conf, iou=iou)


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
