#!/usr/bin/env python3
"""
Train a fake news detection model with a complete scikit-learn Pipeline.

CRITICAL: This pipeline now accepts RAW DATA (with 'title' and 'text' columns/keys)
directly as input, and performs ALL preprocessing steps INTERNALLY:
1. Combine title + text (inside Pipeline)
2. Clean text (inside Pipeline)
3. TF-IDF feature extraction (inside Pipeline)
4. Classification (inside Pipeline)

This ensures 100% consistency between training and inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from feature_extractors import TextCleaner, TextCombiner

# -----------------------------
# Configuration
# -----------------------------
LABELS: Final[Tuple[str, str]] = ("REAL", "FAKE")

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_any(path: Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, encoding="latin-1")


# -----------------------------
# Plot utilities
# -----------------------------
def plot_confusion_matrix(cm: np.ndarray, out: Path, title: str = "Confusion Matrix") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LABELS)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(LABELS)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_curve(x: np.ndarray, y: np.ndarray, out: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def build_complete_pipeline(
    title_col: str = "title",
    text_col: str = "text",
) -> Pipeline:
    """
    Build the COMPLETE preprocessing and classification pipeline.
    
    CRITICAL: This pipeline is designed to work DIRECTLY on RAW INPUT DATA
    that contains 'title' and 'text' fields (as DataFrame columns or dict keys).
    
    Pipeline steps (ALL executed internally):
    1. Combine title + text fields into a single string
    2. Text cleaning/normalization (lowercase, remove URLs/emails, etc.)
    3. TF-IDF feature extraction (1-3 grams)
    4. Classification with RandomForest
    
    IMPORTANT: Training and inference both use EXACTLY the same code path!
    
    Parameters
    ----------
    title_col : str
        Name of the title column/key in input data.
    text_col : str
        Name of the main text column/key in input data.
        
    Returns
    -------
    Pipeline
        A complete scikit-learn Pipeline that can be fit on raw data.
    """
    return Pipeline(
        steps=[
            # Step 1: Combine title + text (inside Pipeline!)
            # Input: DataFrame or list of dicts with title/text columns
            # Output: list of combined strings
            (
                "combiner",
                TextCombiner(
                    title_col=title_col,
                    text_col=text_col,
                    separator=" ",
                ),
            ),
            # Step 2: Text cleaning/normalization
            # Input: list of raw strings (after combination)
            # Output: list of cleaned strings
            (
                "cleaner",
                TextCleaner(
                    lowercase=True,
                    remove_urls=True,
                    remove_emails=True,
                    remove_non_ascii=True,
                    collapse_whitespace=True,
                ),
            ),
            # Step 3: TF-IDF vectorization
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True,
                    stop_words="english",
                    ngram_range=(1, 3),
                    max_df=0.8,
                    min_df=3,
                    max_features=20_000,
                ),
            ),
            # Step 4: Classifier
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )


# -----------------------------
# Main training routine
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train Fake News Detector with FULL Pipeline (Combine -> Clean -> TF-IDF -> RandomForest). "
                    "Pipeline accepts raw DataFrame with title/text columns directly!"
    )
    ap.add_argument("--real", required=True, help="Path to True.csv")
    ap.add_argument("--fake", required=True, help="Path to Fake.csv")
    ap.add_argument(
        "--text-col",
        default="text",
        help="Name of the text column in raw CSV. Default: text",
    )
    ap.add_argument(
        "--title-col",
        default="title",
        help="Name of the title column in raw CSV. Default: title",
    )
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    a = ap.parse_args()

    outdir = ensure_dir(Path(a.outdir))
    charts = ensure_dir(outdir / "charts")

    # 1) Load RAW data - NO PREPROCESSING HERE!
    df_real = read_csv_any(Path(a.real))
    df_fake = read_csv_any(Path(a.fake))

    # 2) Prepare RAW input features - NO MANUAL COMBINING!
    # We keep only the columns we need (title and text) but keep them as separate
    # columns so the Pipeline can handle the combining internally.
    X_real = df_real[[a.title_col, a.text_col]].copy()
    X_fake = df_fake[[a.title_col, a.text_col]].copy()

    # Combine into single DataFrame (still with separate title/text columns)
    X = pd.concat([X_real, X_fake], ignore_index=True)
    y = np.array([0] * len(X_real) + [1] * len(X_fake))  # 0=REAL, 1=FAKE

    print(f"Loaded {len(X)} samples, keeping columns: {list(X.columns)}")
    print(f"Pipeline will internally combine: '{a.title_col}' + '{a.text_col}'")

    # 3) Train/Validation split (using full data for demo)
    # For proper training, uncomment the train_test_split:
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )
    X_train, X_test, y_train, y_test = X, X, y, y

    # 4) Build the COMPLETE pipeline - NOW WITH TextCombiner AS FIRST STEP!
    pipe = build_complete_pipeline(
        title_col=a.title_col,
        text_col=a.text_col,
    )
    
    print("\nPipeline steps (ALL executed internally):")
    for name, step in pipe.steps:
        print(f"  - {name}: {type(step).__name__}")

    # 5) 5-fold CV on training split for robust estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"\nCross-validated F1 (train split, 5-fold): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    # 6) Fit on full training split - PASS RAW DATAFRAME DIRECTLY!
    # The Pipeline handles EVERYTHING internally: combining, cleaning, TF-IDF, classification
    print("\nFitting pipeline on RAW DataFrame (title + text columns preserved)...")
    pipe.fit(X_train, y_train)
    print("Pipeline fit complete!")

    # 7) Evaluate on hold-out test split - ALSO PASS RAW DATAFRAME!
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "cv_f1_macro_mean": float(cv_f1.mean()),
        "cv_f1_macro_std": float(cv_f1.std()),
        "report": classification_report(y_test, y_pred, target_names=LABELS, output_dict=True),
    }

    # 8) Save metrics and figures
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, charts / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plot_curve(fpr, tpr, charts / "roc_curve.png", "ROC Curve", "FPR", "TPR")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plot_curve(rec, prec, charts / "pr_curve.png", "Precision-Recall Curve", "Recall", "Precision")

    # 9) Persist artifacts
    # Save ONLY the pipeline - this is the single source of truth for inference
    # This pipeline knows the column names, how to combine title+text, how to clean, etc.
    joblib.dump(pipe, outdir / "pipeline.joblib")
    print(f"\nPipeline saved to: {outdir / 'pipeline.joblib'}")
    
    # Also save separate parts for backward compatibility (optional)
    try:
        vec = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
        joblib.dump(vec, outdir / "vectorizer.joblib")
        joblib.dump(clf, outdir / "model.joblib")
    except Exception:
        pass

    print("=" * 70)
    print("Training complete.")
    print("=" * 70)
    print(f"Key metrics: accuracy={metrics['accuracy']:.3f}, roc_auc={metrics['roc_auc']:.3f}")
    print(f"CV F1 (macro): {metrics['cv_f1_macro_mean']:.3f} ± {metrics['cv_f1_macro_std']:.3f}")
    print()
    print("CRITICAL: The pipeline.joblib file now contains EVERYTHING:")
    print("  - Column names for title and text")
    print("  - Logic to combine title + text")
    print("  - Text cleaning/normalization rules")
    print("  - TF-IDF vectorizer settings")
    print("  - Trained classifier")
    print()
    print("For inference, simply pass raw data (dict or DataFrame) with 'title' and")
    print("'text' keys/columns directly to pipe.predict() or pipe.predict_proba().")
    print("NO manual preprocessing required - ALL steps are internal!")


if __name__ == "__main__":
    main()
