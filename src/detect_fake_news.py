#!/usr/bin/env python3
"""
Detect fake news using a trained pipeline.

CRITICAL: This script uses the EXACT SAME preprocessing code path as training.
The pipeline expects input data (dict or DataFrame) with 'title' and 'text' keys/columns,
and INTERNALLY performs: title+text combination -> text cleaning -> TF-IDF -> classification.

NO MANUAL PREPROCESSING IS NEEDED - ALL STEPS ARE ENCAPSULATED IN THE PIPELINE!
"""

from __future__ import annotations
import argparse
from pathlib import Path
import joblib
from typing import Optional, Union, Dict, Tuple

import pandas as pd


def load_pipeline(pipeline_path: Union[str, Path]) -> object:
    """
    Load the COMPLETE trained pipeline from disk.
    
    The loaded pipeline contains EVERYTHING needed for inference:
    - Column names for title and text (configured during training)
    - Logic to combine title + text
    - Text cleaning/normalization rules
    - TF-IDF vectorizer (fitted vocabulary)
    - Trained classifier
    
    Parameters
    ----------
    pipeline_path : str or Path
        Path to the saved pipeline.joblib file.
        
    Returns
    -------
    Pipeline
        The loaded scikit-learn Pipeline ready for inference.
        
    Raises
    ------
    FileNotFoundError
        If the pipeline file does not exist.
    ValueError
        If the loaded object is not a valid Pipeline.
    """
    path = Path(pipeline_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline not found at: {path}")
    
    pipe = joblib.load(path)
    
    # Basic validation to ensure we have a proper pipeline
    if not hasattr(pipe, "predict_proba") or not hasattr(pipe, "predict"):
        raise ValueError(
            "Loaded object is not a valid pipeline. "
            "Expected a Pipeline with predict_proba and predict methods."
        )
    
    return pipe


def predict_news(
    pipe: object,
    title: str = "",
    text: str = "",
    threshold: float = 0.40,
    return_probability: bool = False,
) -> Union[str, Tuple[str, float]]:
    """
    Predict if a news article is fake or real using the trained pipeline.
    
    CRITICAL: This function uses the EXACT SAME preprocessing path as training.
    It accepts raw title and text (just like the raw training data), and the
    Pipeline INTERNALLY handles ALL preprocessing steps:
    1. Combine title + text (using same logic as training)
    2. Clean text (using same rules as training: lowercase, remove URLs, etc.)
    3. TF-IDF vectorization (using same vocabulary as training)
    4. Classification (same model as training)
    
    This ensures 100% consistency between training and inference!
    
    Parameters
    ----------
    pipe : Pipeline
        Trained scikit-learn Pipeline (loaded via load_pipeline()).
    title : str
        News article title. Can be empty string if only text is available.
    text : str
        News article body/text. Can be empty string if only title is available.
    threshold : float
        Decision threshold for FAKE classification (default: 0.40).
        A lower threshold increases recall for FAKE news (fewer false negatives),
        while a higher threshold increases precision (fewer false positives).
    return_probability : bool
        If True, return both classification label and FAKE probability.
        
    Returns
    -------
    str or tuple
        Classification label ("FAKE" or "REAL"), or tuple of (label, probability).
        
    Examples
    --------
    >>> pipe = load_pipeline("outputs/pipeline.joblib")
    >>> # Predict with both title and text (same structure as training data!)
    >>> label, prob = predict_news(pipe, 
    ...                            title="Breaking News",
    ...                            text="Article body content here...",
    ...                            return_probability=True)
    >>> print(f"{label} (probability: {prob:.3f})")
    
    >>> # Predict with only text (title defaults to empty string)
    >>> label = predict_news(pipe, text="Article text only...")
    """
    # Create input in the SAME STRUCTURE as training data
    # The Pipeline expects a DataFrame (or list of dicts) with 'title' and 'text' columns
    input_data = pd.DataFrame([{
        "title": title,
        "text": text,
    }])
    
    # Run prediction - PIPELINE HANDLES EVERYTHING INTERNALLY!
    # No manual cleaning, no manual combining - it's all encapsulated!
    prob = float(pipe.predict_proba(input_data)[0, 1])
    label = "FAKE" if prob >= threshold else "REAL"
    
    if return_probability:
        return label, prob
    return label


def predict_news_batch(
    pipe: object,
    articles: list[Dict[str, str]],
    threshold: float = 0.40,
) -> list[Tuple[str, float]]:
    """
    Predict multiple news articles in batch.
    
    Parameters
    ----------
    pipe : Pipeline
        Trained scikit-learn Pipeline.
    articles : list of dict
        List of article dicts, each with 'title' and 'text' keys.
    threshold : float
        Decision threshold for FAKE classification.
        
    Returns
    -------
    list of tuples
        List of (label, probability) tuples for each article.
    """
    # Convert to DataFrame for batch prediction
    input_data = pd.DataFrame(articles)
    
    # Batch prediction (still uses EXACT same pipeline!)
    probs = pipe.predict_proba(input_data)[:, 1]
    labels = ["FAKE" if p >= threshold else "REAL" for p in probs]
    
    return list(zip(labels, probs))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect fake news using a COMPLETE trained pipeline. "
                    "The pipeline uses EXACTLY the same preprocessing as training: "
                    "title+text combination -> text cleaning -> TF-IDF -> classification. "
                    "NO manual preprocessing is performed in this script!"
    )
    ap.add_argument(
        "--pipeline", 
        required=True, 
        help="Path to pipeline.joblib (contains ALL logic: combining, cleaning, TF-IDF, model)."
    )
    ap.add_argument(
        "--title", 
        default="",
        help="News article title (recommended, but optional)."
    )
    ap.add_argument(
        "--text", 
        default="",
        help="News article text/body (if neither title nor text is provided, this is required)."
    )
    ap.add_argument(
        "--threshold", 
        type=float, 
        default=0.40,
        help="Decision threshold for FAKE classification (default: 0.40). "
             "Lower values = more sensitive to FAKE news (higher recall); "
             "Higher values = more conservative (higher precision)."
    )
    ap.add_argument(
        "--verbose", 
        action="store_true",
        help="Print verbose output including pipeline structure."
    )
    args = ap.parse_args()

    # Validate input
    if not args.title.strip() and not args.text.strip():
        print("Error: At least one of --title or --text must be provided.")
        exit(1)

    try:
        # Load the COMPLETE pipeline - this is the ONLY artifact needed!
        pipe = load_pipeline(args.pipeline)
        
        if args.verbose:
            print("=" * 60)
            print("Pipeline Configuration (ALL steps executed internally):")
            print("=" * 60)
            for name, step in pipe.steps:
                print(f"  Step: {name}")
                print(f"    Type: {type(step).__name__}")
                if hasattr(step, 'get_params'):
                    params = step.get_params()
                    # Show relevant params based on step type
                    if name == 'combiner':
                        print(f"    title_col: '{params.get('title_col', 'title')}'")
                        print(f"    text_col: '{params.get('text_col', 'text')}'")
                    elif name == 'cleaner':
                        print(f"    lowercase: {params.get('lowercase', True)}")
                        print(f"    remove_urls: {params.get('remove_urls', True)}")
            print("=" * 60)
            print()
        
        # Run prediction - use EXACTLY the same structure as training input!
        # Raw title and text go in, prediction comes out - ALL work done internally by Pipeline
        print(f"Input structure matches training: 'title' + 'text' fields")
        print(f"Processing with Pipeline (combine -> clean -> TF-IDF -> classify)...")
        print()
        
        label, prob = predict_news(
            pipe, 
            title=args.title,
            text=args.text,
            threshold=args.threshold,
            return_probability=True,
        )
        
        # Output results
        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Classification: {label}")
        print(f"FAKE Probability: {prob:.1%}")
        print(f"Threshold Used: {args.threshold:.2f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
