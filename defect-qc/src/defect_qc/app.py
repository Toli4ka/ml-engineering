from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from omegaconf import OmegaConf
from PIL import Image

from defect_qc.data import build_dataframes, create_manifest, get_data_dir
from defect_qc.inference import (
    DemoModel,
    load_demo_model,
    make_gradcam_overlay,
    predict_image,
    predict_tensor,
    preprocess_image,
)


LABEL_NAMES = {0: "ok", 1: "defect"}
DEFAULT_CHECKPOINT = Path("artifacts/demo_model.pt")


st.set_page_config(page_title="Defect QC Demo", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; max-width: 1180px; }
    .decision {
        border-radius: 8px;
        padding: 18px 20px;
        font-size: 1.55rem;
        font-weight: 750;
        letter-spacing: 0;
        text-align: center;
        margin: 0.25rem 0 1rem 0;
    }
    .decision-ok {
        background: #e9f7ef;
        border: 1px solid #8bd3a5;
        color: #115d2e;
    }
    .decision-defect {
        background: #ffecea;
        border: 1px solid #f09891;
        color: #8f1d14;
    }
    .small-note {
        color: #5c6570;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_model(checkpoint_path: str) -> DemoModel:
    return load_demo_model(checkpoint_path)


@st.cache_data(show_spinner=False)
def cached_manifest() -> pd.DataFrame:
    data_dir = get_data_dir()
    df = create_manifest(data_dir)
    cfg = OmegaConf.create({
        "data": {
            "val_size": 0.2,
            "random_state": 16,
            "label_col": "defect",
        }
    })
    df_train, df_val, df_test = build_dataframes(cfg, df)
    return pd.concat([df_train, df_val, df_test], ignore_index=True)


def load_sample_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.copy()


def outcome_name(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "True positive"
    if y_true == 0 and y_pred == 0:
        return "True negative"
    if y_true == 0 and y_pred == 1:
        return "False positive"
    return "False negative"


def render_decision(label_idx: int) -> None:
    if label_idx == 1:
        css_class = "decision decision-defect"
        text = "REJECT / DEFECT"
    else:
        css_class = "decision decision-ok"
        text = "PASS / OK"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def render_prediction_panel(
    demo_model: DemoModel,
    image: Image.Image,
    threshold: float,
    y_true: int | None,
) -> None:
    prediction, x = predict_image(demo_model, image, threshold)
    render_decision(prediction.label_idx)

    metric_cols = st.columns(3)
    metric_cols[0].metric("P(defect)", f"{prediction.prob_defect:.3f}")
    metric_cols[1].metric("P(ok)", f"{prediction.prob_ok:.3f}")
    metric_cols[2].metric("Threshold", f"{threshold:.2f}")

    st.progress(prediction.prob_defect, text="Defect probability")

    if y_true is not None:
        st.write(
            f"True label: **{LABEL_NAMES[y_true]}** | "
            f"Prediction: **{prediction.label}** | "
            f"Outcome: **{outcome_name(y_true, prediction.label_idx)}**"
        )

    st.markdown("**Model Evidence**")
    overlay = make_gradcam_overlay(demo_model, x, class_idx=1)
    st.image(overlay, caption="Grad-CAM evidence, not a defect segmentation mask")


@st.cache_data(show_spinner=False)
def batch_predictions(checkpoint_path: str, threshold: float, split: str) -> pd.DataFrame:
    demo_model = load_demo_model(checkpoint_path, device=torch.device("cpu"))
    df = cached_manifest()
    rows = []
    for row in df[df["split"] == split].itertuples(index=False):
        image = load_sample_image(row.file_path)
        x = preprocess_image(image, demo_model.img_mode, demo_model.img_size)
        prediction = predict_tensor(demo_model, x, threshold)
        rows.append(
            {
                "file_path": row.file_path,
                "y_true": int(row.defect),
                "y_pred": prediction.label_idx,
                "prob_defect": prediction.prob_defect,
            }
        )
    return pd.DataFrame(rows)


def render_batch_panel(checkpoint_path: str, threshold: float) -> None:
    st.subheader("Threshold Behavior")
    split = st.selectbox("Split", ["test", "val", "train"], index=0)
    if not st.button("Evaluate split"):
        st.markdown('<p class="small-note">Run this after choosing a threshold.</p>', unsafe_allow_html=True)
        return

    with st.spinner("Scoring images..."):
        scored = batch_predictions(checkpoint_path, threshold, split)

    y_true = scored["y_true"]
    y_pred = scored["y_pred"]
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    cols = st.columns(4)
    cols[0].metric("Defects rejected", tp)
    cols[1].metric("OK accepted", tn)
    cols[2].metric("False alarms", fp)
    cols[3].metric("Missed defects", fn)

    cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["true ok", "true defect"],
        columns=["pred ok", "pred defect"],
    )
    st.dataframe(cm, use_container_width=True)

    mistakes = scored[scored["y_true"] != scored["y_pred"]].copy()
    if mistakes.empty:
        st.success("No mistakes at this threshold.")
    else:
        mistakes["true_label"] = mistakes["y_true"].map(LABEL_NAMES)
        mistakes["pred_label"] = mistakes["y_pred"].map(LABEL_NAMES)
        st.dataframe(
            mistakes[["true_label", "pred_label", "prob_defect", "file_path"]]
            .sort_values("prob_defect", ascending=False)
            .head(20),
            use_container_width=True,
        )


def main() -> None:
    st.title("Casting Defect QC")
    st.caption("Inspect a casting image, tune the defect threshold, and review the model evidence.")

    with st.sidebar:
        st.header("Model")
        checkpoint_path = st.text_input("Checkpoint", str(DEFAULT_CHECKPOINT))
        threshold = st.slider("Defect threshold", 0.0, 1.0, 0.5, 0.01)
        st.markdown(
            '<p class="small-note">Higher threshold means fewer rejects and more risk of missed defects.</p>',
            unsafe_allow_html=True,
        )

    try:
        demo_model = cached_model(checkpoint_path)
    except FileNotFoundError:
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load checkpoint: {exc}")
        st.stop()

    image: Image.Image | None = None
    y_true: int | None = None

    try:
        df = cached_manifest()
    except Exception as exc:
        st.warning(f"Dataset samples are unavailable: {exc}")
        df = pd.DataFrame()

    if not df.empty:
        split = st.selectbox("Dataset split", ["test", "val", "train"], index=0)
        label_filter = st.selectbox("True label", ["all", "ok", "defect"], index=0)
        sample_df = df[df["split"] == split].copy()
        if label_filter != "all":
            sample_df = sample_df[sample_df["defect"] == int(label_filter == "defect")]

        if sample_df.empty:
            st.warning("No samples match this filter.")
            st.stop()

        current_path = st.session_state.get("sample_row", {}).get("file_path")
        current_is_valid = current_path in set(sample_df["file_path"])
        random_clicked = st.button("Random sample")
        if not current_is_valid or random_clicked:
            st.session_state.sample_row = sample_df.sample(n=1, random_state=None).iloc[0].to_dict()

        row = st.session_state.sample_row
        image = load_sample_image(row["file_path"])
        y_true = int(row["defect"])
        st.markdown(f"Sample: `{Path(row['file_path']).name}`")

    if image is None:
        st.info("Choose a dataset sample.")
        st.stop()

    left, right = st.columns([1, 1.05], gap="large")
    with left:
        st.subheader("Input")
        st.image(image, caption="Original image", use_container_width=True)

    with right:
        st.subheader("Inspection Result")
        render_prediction_panel(demo_model, image, threshold, y_true)

    render_batch_panel(checkpoint_path, threshold)


if __name__ == "__main__":
    main()
