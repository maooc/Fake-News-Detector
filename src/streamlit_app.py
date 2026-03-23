#!/usr/bin/env python3
"""
Streamlit Web Application for Fake News Detection.

This app uses the complete trained pipeline, which handles all feature
engineering internally. No manual text cleaning is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import streamlit as st


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths() -> Dict[str, Path]:
    root = project_root()
    out = root / "outputs"
    return {
        "pipeline": out / "pipeline.joblib",
    }


def load_pipeline(pipeline_path: Path):
    if pipeline_path.exists():
        return joblib.load(pipeline_path)
    return None


def predict(
    pipeline,
    text: str,
    title: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    if title:
        df = pd.DataFrame({"title": [title], "text": [text]})
        prob = float(pipeline.predict_proba(df)[0, 1])
    else:
        prob = float(pipeline.predict_proba([text])[0, 1])

    label = "FAKE" if prob >= threshold else "REAL"
    confidence = max(prob, 1 - prob)

    return {
        "label": label,
        "probability": prob,
        "confidence": confidence,
    }


def main():
    dp = default_paths()

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pipeline", default=str(dp["pipeline"]))
    args, _ = ap.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()

    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="centered",
    )

    st.title("📰 Fake News & Misinformation Detector")
    st.caption("TF-IDF + RandomForest with complete feature engineering pipeline")

    with st.sidebar:
        st.subheader("Model Configuration")
        st.code(f"Pipeline: {pipeline_path}")
        st.write(f"Pipeline exists: **{pipeline_path.exists()}**")

        if not pipeline_path.exists():
            st.warning(
                "Pipeline not found. Please train the model first:\n\n"
                "```\npython src/train_model.py --real data/True.csv --fake data/Fake.csv\n```"
            )

    pipeline = load_pipeline(pipeline_path)
    if pipeline is None:
        st.error(
            "Model artifacts not found.\n\n"
            "Please run training first:\n\n"
            "```\npython src/train_model.py --real data/True.csv --fake data/Fake.csv\n```\n\n"
            "This will create `outputs/pipeline.joblib` with all feature engineering steps."
        )
        st.stop()

    st.subheader("Input")
    title = st.text_input("Title (optional)", placeholder="Enter article headline...")
    text = st.text_area(
        "Article Text",
        height=200,
        placeholder="Paste the article content here...",
    )

    threshold = st.slider(
        "FAKE decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Lower values make the model more sensitive to detecting fake news",
    )

    if st.button("Analyze", type="primary") and text.strip():
        with st.spinner("Analyzing..."):
            result = predict(
                pipeline,
                text=text,
                title=title if title.strip() else None,
                threshold=threshold,
            )

        st.subheader("Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Prediction",
                result["label"],
                delta="Likely fake" if result["label"] == "FAKE" else "Likely real",
            )
        with col2:
            st.metric(
                "Confidence",
                f"{result['confidence']:.1%}",
            )

        st.progress(
            result["probability"] if result["label"] == "FAKE" else 1 - result["probability"],
            text=f"Fake probability: {result['probability']:.1%} (threshold: {threshold:.2f})",
        )

        with st.expander("Pipeline Details"):
            st.write("**Pipeline Steps:**")
            for name, step in pipeline.named_steps.items():
                st.write(f"- {name}: `{type(step).__name__}`")


if __name__ == "__main__":
    main()
