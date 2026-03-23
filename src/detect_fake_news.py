#!/usr/bin/env python3
"""
Fake News Detection Inference Script.

This script provides a simplified interface for predicting whether a given
text is real or fake news. All feature engineering steps are encapsulated
in the saved pipeline, ensuring consistency with training.

Usage:
    python detect_fake_news.py --pipeline outputs/pipeline.joblib --text "Your news text here"

The pipeline handles all preprocessing automatically:
- Text cleaning (lowercase, URL removal, etc.)
- Feature combination (if title+text provided)
- TF-IDF vectorization
- Classification
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_pipeline(pipeline_path: str | Path) -> Pipeline:
    """
    Load a trained pipeline from disk.

    Parameters
    ----------
    pipeline_path : str | Path
        Path to the saved pipeline.joblib file.

    Returns
    -------
    Pipeline
        Loaded scikit-learn pipeline with all feature engineering steps.
    """
    path = Path(pipeline_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline not found: {path}")
    return joblib.load(path)


def predict_single(
    pipeline: Pipeline,
    text: str,
    *,
    title: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Predict whether a single text is fake or real news.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline with feature engineering and classifier.
    text : str
        The article text to classify.
    title : str, optional
        Optional title to combine with text.
    threshold : float
        Decision threshold for FAKE classification.

    Returns
    -------
    dict
        Prediction results with keys:
        - label: "FAKE" or "REAL"
        - probability: float, probability of being fake
        - confidence: float, confidence score
    """
    if title is not None:
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
        "threshold": threshold,
    }


def predict_batch(
    pipeline: Pipeline,
    texts: list[str],
    *,
    titles: Optional[list[str]] = None,
    threshold: float = 0.5,
) -> list[Dict[str, Any]]:
    """
    Predict labels for multiple texts.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline.
    texts : list[str]
        List of article texts.
    titles : list[str], optional
        Optional list of titles.
    threshold : float
        Decision threshold.

    Returns
    -------
    list[dict]
        List of prediction results.
    """
    if titles is not None:
        df = pd.DataFrame({"title": titles, "text": texts})
        probs = pipeline.predict_proba(df)[:, 1]
    else:
        probs = pipeline.predict_proba(texts)[:, 1]

    results = []
    for prob in probs:
        label = "FAKE" if prob >= threshold else "REAL"
        confidence = max(prob, 1 - prob)
        results.append({
            "label": label,
            "probability": float(prob),
            "confidence": float(confidence),
            "threshold": threshold,
        })

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect fake news using a trained pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with just text
  python detect_fake_news.py --pipeline outputs/pipeline.joblib --text "Breaking news..."

  # With title and custom threshold
  python detect_fake_news.py --pipeline outputs/pipeline.joblib --title "Headline" --text "Article content..." --threshold 0.4
        """,
    )
    ap.add_argument(
        "--pipeline",
        required=True,
        help="Path to the trained pipeline.joblib",
    )
    ap.add_argument(
        "--text",
        required=True,
        help="Article text to classify",
    )
    ap.add_argument(
        "--title",
        help="Optional article title (will be combined with text)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for FAKE (default: 0.5)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    args = ap.parse_args()

    pipeline = load_pipeline(args.pipeline)

    result = predict_single(
        pipeline,
        text=args.text,
        title=args.title,
        threshold=args.threshold,
    )

    if args.verbose:
        print("=" * 50)
        print("Fake News Detection Result")
        print("=" * 50)
        print(f"Label:      {result['label']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Confidence:  {result['confidence']:.4f}")
        print(f"Threshold:   {result['threshold']:.2f}")
        print("=" * 50)
    else:
        print(f"Label: {result['label']} | Fake probability: {result['probability']:.3f}")


if __name__ == "__main__":
    main()
