from pathlib import Path

import mlflow.sklearn
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf


PROJECT_DIR = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_DIR / "configs" / "config.yaml"
MLRUNS_DIR = PROJECT_DIR / "mlruns"


@st.cache_data
def load_config():
    return OmegaConf.load(CONFIG_PATH)


def find_latest_model_dir() -> Path | None:
    model_files = list(MLRUNS_DIR.glob("*/models/*/artifacts/MLmodel"))
    if not model_files:
        return None
    return max(model_files, key=lambda path: path.stat().st_mtime).parent


@st.cache_resource
def load_model(model_dir: str):
    return mlflow.sklearn.load_model(model_dir)


def build_input_row(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    embarked: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Embarked": embarked,
            }
        ]
    )


def main():
    st.set_page_config(page_title="Titanic Survival Predictor")

    cfg = load_config()
    threshold = float(cfg.eval.threshold)
    model_dir = find_latest_model_dir()

    st.title("Titanic Survival Predictor  🚢")
    st.write("Enter passenger details and estimate the chance of survival.")

    if model_dir is None:
        st.error("No trained model was found. Run `python src/train.py` first.")
        st.stop()

    model = load_model(str(model_dir))

    st.sidebar.header("Model")
    st.sidebar.write(f"Decision threshold: `{threshold:.2f}`")
    st.sidebar.write(f"Loaded artifact: `{model_dir.relative_to(PROJECT_DIR)}`")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pclass = st.selectbox(
                "Passenger class",
                options=[1, 2, 3],
                format_func=lambda value: f"{value}",
                index=2,
            )
            sex = st.selectbox("Sex", options=["female", "male"], index=1)
            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
            )

        with col2:
            sibsp = st.number_input(
                "Siblings/spouses aboard",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )
            parch = st.number_input(
                "Parents/children aboard",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
            )
            embarked = st.selectbox(
                "Port of embarkation",
                options=["C", "Q", "S"],
                format_func=lambda value: {
                    "C": "Cherbourg",
                    "Q": "Queenstown",
                    "S": "Southampton",
                }[value],
                index=2,
            )

        submitted = st.form_submit_button("Predict")

    if not submitted:
        st.info("Complete the fields and run a prediction.")
        return

    input_row = build_input_row(pclass, sex, age, sibsp, parch, embarked)
    survival_probability = float(model.predict_proba(input_row)[0, 1])
    survived = survival_probability >= threshold

    st.subheader("Prediction")
    st.metric("Survival probability", f"{survival_probability:.1%}")

    if survived:
        st.success("Predicted result: Survived 🎉")
    else:
        st.warning("Predicted result: Did not survive 🙉")

    with st.expander("Input sent to the model"):
        st.dataframe(input_row, hide_index=True)


if __name__ == "__main__":
    main()
