#!/usr/bin/env python3
"""
Streamlit web app for fake news detection.

CRITICAL: This app uses the EXACT SAME preprocessing pipeline as training.
Raw user input (title + text) is passed directly to the Pipeline, which
internally performs: title+text combination -> text cleaning -> TF-IDF -> classification.

NO manual preprocessing in this app - ALL logic is encapsulated in the Pipeline!
"""

import argparse
from pathlib import Path
import streamlit as st

# Import from our refactored modules (single source of truth!)
from detect_fake_news import load_pipeline, predict_news

# ---------- path helpers ----------
def project_root() -> Path:
    # this file is in src/, project root is parent directory
    return Path(__file__).resolve().parents[1]

def default_pipeline_path():
    root = project_root()
    out = root / "outputs"
    return out / "pipeline.joblib"

# ---------- streamlit app ----------
def main():
    # parse CLI overrides but give safe defaults relative to repo root
    default_path = default_pipeline_path()

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pipeline", default=str(default_path))
    args, _ = ap.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()

    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
    st.title("📰 Fake News & Misinformation Detector")
    st.caption("Unified Pipeline: Combine → Clean → TF-IDF → RandomForest")

    # sidebar: show where we look for files
    with st.sidebar:
        st.subheader("Model Artifacts")
        st.code(f"pipeline:  {pipeline_path}")
        st.write(f"Exists → pipeline: **{pipeline_path.exists()}**")
        
        st.markdown("---")
        st.subheader("About")
        st.info(
            "This app uses a **unified scikit-learn Pipeline** that contains "
            "ALL preprocessing and classification logic in a single artifact. "
            "The EXACT same code path is used for both training and inference, "
            "guaranteeing consistent results.\n\n"
            "Pipeline steps (ALL internal):\n"
            "1. Combine title + text\n"
            "2. Clean/normalize text\n"
            "3. TF-IDF feature extraction\n"
            "4. RandomForest classification"
        )

    # Load pipeline (only ONE artifact needed now!)
    try:
        pipe = load_pipeline(pipeline_path)
    except Exception as e:
        st.error(
            f"Model artifact not found or invalid.\n\n"
            f"Error: {str(e)}\n\n"
            f"• Ensure you trained and saved the pipeline to `outputs/`\n"
            f"• Run the training script first: `python src/train_model.py`\n"
            f"• Or run Streamlit with explicit path, e.g.:\n"
            f"  `streamlit run src/streamlit_app.py -- --pipeline /path/to/pipeline.joblib`"
        )
        st.stop()

    # Show pipeline information
    with st.expander("Pipeline Details (Click to expand)"):
        st.write("Current pipeline configuration (steps executed in order):")
        for i, (name, step) in enumerate(pipe.steps, 1):
            params = step.get_params() if hasattr(step, 'get_params') else {}
            
            # Build step description
            desc = f"**Step {i}: {name}** (`{type(step).__name__}`)"
            
            # Add relevant details
            details = []
            if name == 'combiner':
                details.append(f"  • Columns: '{params.get('title_col', 'title')}' + '{params.get('text_col', 'text')}'")
            elif name == 'cleaner':
                details.append(f"  • lowercase: {params.get('lowercase', True)}")
                details.append(f"  • remove_urls: {params.get('remove_urls', True)}")
                details.append(f"  • remove_emails: {params.get('remove_emails', True)}")
            elif name == 'tfidf':
                details.append(f"  • ngram_range: {params.get('ngram_range', '(1,1)')}")
                details.append(f"  • max_features: {params.get('max_features', 'N/A')}")
            elif name == 'clf':
                details.append(f"  • n_estimators: {params.get('n_estimators', 'N/A')}")
            
            st.markdown(desc)
            for d in details:
                st.markdown(d)
            st.markdown("")

    # User input - matches TRAINING DATA STRUCTURE: title + text
    st.subheader("Enter News Article")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Optional: provide sample inputs
        sample_type = st.selectbox(
            "Load sample:",
            ["Custom Input", "Sample Real News", "Sample Fake News"]
        )
    
    # Sample inputs matching training data structure
    sample_real = {
        "title": "Scientists Discover New Species in Deep Ocean Expedition",
        "text": "Marine biologists have identified over 100 previously unknown species during a groundbreaking deep-sea exploration mission. The research team used advanced submersible technology to reach depths never before studied, revealing a diverse ecosystem of unique organisms adapted to extreme pressure and darkness."
    }
    sample_fake = {
        "title": "SHOCKING: Government Admits Hiding Alien Technology for 50 Years",
        "text": "YOU WON'T BELIEVE what whistleblowers have revealed! Top-secret documents confirm the government has been reverse-engineering extraterrestrial technology in hidden underground bases. Sources say this technology could provide FREE ENERGY to the entire world but is being suppressed by BIG OIL! SHARE THIS BEFORE THEY CENSOR IT!!!"
    }
    
    # Initialize with empty or sample values
    if sample_type == "Sample Real News":
        init_title, init_text = sample_real["title"], sample_real["text"]
    elif sample_type == "Sample Fake News":
        init_title, init_text = sample_fake["title"], sample_fake["text"]
    else:
        init_title, init_text = "", ""
    
    # User input fields (same structure as training data: title + text)
    news_title = st.text_input("Article Title (optional but recommended):", value=init_title)
    news_text = st.text_area("Article Text:", value=init_text, height=200)
    
    # Threshold slider
    threshold = st.slider(
        "FAKE decision threshold", 
        0.05, 0.95, 0.40, 0.01,
        help="Lower values = more sensitive to FAKE news (higher recall)\n"
             "Higher values = more conservative (higher precision)\n"
             "Default: 0.40 (optimized for fake news recall)"
    )

    # Analyze button
    if st.button("Analyze", type="primary") and (news_title.strip() or news_text.strip()):
        with st.spinner("Analyzing with unified pipeline (combine → clean → vectorize → classify)..."):
            # CRITICAL: Use EXACT SAME INPUT STRUCTURE as TRAINING!
            # Raw title and text go directly to pipeline - NO manual preprocessing!
            # The Pipeline handles EVERYTHING internally, guaranteeing consistency.
            label, prob_fake = predict_news(
                pipe, 
                title=news_title,
                text=news_text,
                threshold=threshold,
                return_probability=True,
            )

        # Display results
        st.subheader("Analysis Result")
        
        # Show what was processed
        with st.expander("Input sent to pipeline"):
            st.json({
                "title": news_title[:200] + ("..." if len(news_title) > 200 else ""),
                "text": news_text[:300] + ("..." if len(news_text) > 300 else "")
            })
        
        # Classification result
        if label == "FAKE":
            st.error(f"### Classification: {label} 🚩")
        else:
            st.success(f"### Classification: {label} ✓")
            
        st.metric("FAKE Probability", f"{prob_fake:.1%}")
        
        # Progress bar visualization
        if label == "FAKE":
            st.progress(prob_fake)
        else:
            st.progress(1 - prob_fake)
            
        st.caption(f"Threshold used: {threshold:.2f}")
        
        # Add interpretation
        with st.expander("Interpretation"):
            if prob_fake < 0.2:
                st.info("This content has characteristics strongly associated with real news.")
            elif prob_fake < 0.4:
                st.info("This content leans toward real but has some mixed signals.")
            elif prob_fake < 0.6:
                st.warning("This content has mixed characteristics - exercise caution and verify sources.")
            elif prob_fake < 0.8:
                st.warning("This content has characteristics commonly associated with fake or misleading news.")
            else:
                st.error("This content has strong characteristics associated with fake news or misinformation.")


if __name__ == "__main__":
    main()
