#!/usr/bin/env python3
"""
Fake News Detection Model Training Script.

This script trains a fake news classifier using a complete scikit-learn Pipeline
that includes all feature engineering steps, ensuring consistency between
training and inference.

Pipeline Architecture:
1. FeatureEngineer: Combines title+text and cleans text
2. TfidfVectorizer: Converts text to TF-IDF features
3. RandomForestClassifier: Classification model

Usage:
    python train_model.py --real data/True.csv --fake data/Fake.csv --outdir outputs
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

from feature_engineering import FeatureEngineer

LABELS: Final[Tuple[str, str]] = ("REAL", "FAKE")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_any(path: Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=nrows, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, encoding="latin-1")


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


def build_pipeline(
    *,
    title_col: str = "title",
    text_col: str = "text",
    max_features: int = 20_000,
    n_estimators: int = 400,
    random_state: int = 42,
) -> Pipeline:
    """
    Build the complete ML pipeline with feature engineering.

    The pipeline includes:
    1. FeatureEngineer: Combines title+text and applies text cleaning
    2. TfidfVectorizer: Text to TF-IDF features (1-3 grams)
    3. RandomForestClassifier: Final classifier

    Parameters
    ----------
    title_col : str
        Name of the title column.
    text_col : str
        Name of the text column.
    max_features : int
        Maximum TF-IDF features.
    n_estimators : int
        Number of trees in RandomForest.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Complete scikit-learn pipeline.
    """
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FeatureEngineer(
                    title_col=title_col,
                    text_col=text_col,
                    lowercase=True,
                    remove_urls=True,
                    remove_emails=True,
                    remove_non_ascii=True,
                    collapse_whitespace=True,
                ),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True,
                    stop_words="english",
                    ngram_range=(1, 3),
                    max_df=0.8,
                    min_df=3,
                    max_features=max_features,
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train Fake News Detector with complete feature engineering pipeline"
    )
    ap.add_argument("--real", required=True, help="Path to True.csv (real news)")
    ap.add_argument("--fake", required=True, help="Path to Fake.csv (fake news)")
    ap.add_argument(
        "--text-col",
        default="text",
        help="Name of the text column (default: text)",
    )
    ap.add_argument(
        "--title-col",
        default="title",
        help="Name of the title column (default: title)",
    )
    ap.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory for model artifacts",
    )
    ap.add_argument(
        "--max-features",
        type=int,
        default=20_000,
        help="Maximum TF-IDF features (default: 20000)",
    )
    ap.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees in RandomForest (default: 400)",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    a = ap.parse_args()

    outdir = ensure_dir(Path(a.outdir))
    charts = ensure_dir(outdir / "charts")

    print("=" * 60)
    print("Fake News Detector - Model Training")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    df_real = read_csv_any(Path(a.real))
    df_fake = read_csv_any(Path(a.fake))
    print(f"  - Real news samples: {len(df_real)}")
    print(f"  - Fake news samples: {len(df_fake)}")

    print("\n[2/6] Preparing dataset...")
    df_real["label"] = 0
    df_fake["label"] = 1
    df = pd.concat([df_real, df_fake], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df
    y = df["label"].values

    print("\n[3/6] Building pipeline...")
    pipe = build_pipeline(
        title_col=a.title_col,
        text_col=a.text_col,
        max_features=a.max_features,
        n_estimators=a.n_estimators,
    )

    print("\n[4/6] Performing cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    print(f"  - Cross-validated F1 (5-fold): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    print("\n[5/6] Training final model on full dataset...")
    pipe.fit(X, y)

    y_prob = pipe.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "avg_precision": float(average_precision_score(y, y_prob)),
        "cv_f1_macro_mean": float(cv_f1.mean()),
        "cv_f1_macro_std": float(cv_f1.std()),
        "report": classification_report(y, y_pred, target_names=LABELS, output_dict=True),
        "pipeline_steps": list(pipe.named_steps.keys()),
    }

    print("\n[6/6] Saving artifacts...")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y, y_pred)
    plot_confusion_matrix(cm, charts / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y, y_prob)
    plot_curve(fpr, tpr, charts / "roc_curve.png", "ROC Curve", "FPR", "TPR")

    prec, rec, _ = precision_recall_curve(y, y_prob)
    plot_curve(rec, prec, charts / "pr_curve.png", "Precision-Recall Curve", "Recall", "Precision")

    joblib.dump(pipe, outdir / "pipeline.joblib")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nMetrics:")
    print(f"  - Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  - ROC AUC:       {metrics['roc_auc']:.4f}")
    print(f"  - Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"  - CV F1 (mean):  {metrics['cv_f1_macro_mean']:.4f}")
    print(f"\nArtifacts saved to: {outdir.resolve()}")
    print(f"  - pipeline.joblib (complete pipeline with feature engineering)")
    print(f"  - metrics.json")
    print(f"  - charts/")


if __name__ == "__main__":
    main()
