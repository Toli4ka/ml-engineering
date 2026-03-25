from pathlib import Path

import streamlit as st
from PIL import Image

from pedestrian_box.data import list_dataset_images, read_yolo_labels
from pedestrian_box.inference import extract_predictions, run_inference
from pedestrian_box.model import get_default_model_path
from pedestrian_box.visualize import draw_detections


def show_debug_value(enabled, label, value):
    if enabled:
        st.write(f"{label}:", value)


def go_to_next_image():
    total_images = len(st.session_state.image_names)
    st.session_state.selected_image_index = (
        st.session_state.selected_image_index + 1
    ) % total_images


def main():
    st.set_page_config(page_title="Pedestrian Box", layout="wide")
    st.title("Pedestrian Detection Viewer")

    project_root = Path(__file__).resolve().parents[3]
    default_data_config = project_root / "data" / "data.yml"
    default_model_path = get_default_model_path()

    with st.sidebar:
        st.header("Settings")
        data_config_path = Path(default_data_config)
        st.write("Model: YOLO26n")
        st.write("Dataset: FudanPed")
        model_path = Path(default_model_path)
        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU threshold", 0.0, 1.0, 0.45, 0.05)
        debug_mode = st.checkbox("Debug mode", value=False)

    try:
        image_paths = list_dataset_images(data_config_path=data_config_path)
    except Exception as exc:
        st.error(f"Failed to load dataset images: {exc}")
        if debug_mode:
            st.exception(exc)
        return

    if not image_paths:
        st.warning("No images found in train/val folders.")
        return

    image_options = {path.name: path for path in image_paths}
    image_names = list(image_options.keys())
    st.session_state.image_names = image_names

    if "selected_image_index" not in st.session_state:
        st.session_state.selected_image_index = 0

    if st.session_state.selected_image_index >= len(image_names):
        st.session_state.selected_image_index = 0

    selected_name = st.selectbox(
        "Dataset image",
        options=image_names,
        index=st.session_state.selected_image_index,
    )
    st.session_state.selected_image_index = image_names.index(selected_name)
    st.button("Next", on_click=go_to_next_image)
    selected_image_path = image_options[selected_name]
    show_debug_value(debug_mode, "Selected image path", selected_image_path)
    show_debug_value(debug_mode, "Model path", model_path)
    show_debug_value(debug_mode, "Dataset config", data_config_path)

    try:
        with Image.open(selected_image_path) as image:
            original_image = image.convert("RGB")
    except Exception as exc:
        st.error(f"Failed to open image: {exc}")
        if debug_mode:
            st.exception(exc)
        return

    try:
        ground_truth = read_yolo_labels(selected_image_path, data_config_path=data_config_path)
    except Exception as exc:
        st.error(f"Failed to read labels: {exc}")
        if debug_mode:
            st.exception(exc)
        return

    try:
        result = run_inference(
            image_path=selected_image_path,
            model_path=model_path,
            conf=conf_threshold,
            iou=iou_threshold,
        )
        predictions = extract_predictions(result)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        if debug_mode:
            st.exception(exc)
        return

    show_debug_value(debug_mode, "Number of predictions", len(predictions))
    if debug_mode and predictions:
        st.write("Predictions preview:", predictions[:3])

    predicted_image = draw_detections(original_image, predictions, color=(255, 80, 80))
    ground_truth_image = draw_detections(
        original_image,
        [
            {
                "class_id": label["class_id"],
                "class_name": "person",
                "confidence": None,
                "bbox_xyxy": label["bbox_xyxy"],
            }
            for label in ground_truth
        ],
        color=(0, 170, 255),
    )

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Prediction")
        st.image(predicted_image, use_container_width=True)
        st.caption(f"Predicted boxes: {len(predictions)}")
        st.dataframe(predictions, use_container_width=True)

    with right_col:
        st.subheader("Ground truth")
        st.image(ground_truth_image, use_container_width=True)
        st.caption(f"Ground-truth boxes: {len(ground_truth)}")
        st.dataframe(ground_truth, use_container_width=True)


if __name__ == "__main__":
    main()
