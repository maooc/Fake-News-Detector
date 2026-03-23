#!/usr/bin/env python3
"""
Feature Engineering Module for Fake News Detection.

This module provides scikit-learn compatible Transformers for text preprocessing,
ensuring consistent data flow between training and inference pipelines.

Key Components:
- TextCleaner: Text normalization (lowercase, URL removal, etc.)
- TextCombiner: Combines title and text columns
- FeatureEngineer: Complete feature engineering pipeline
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
EXTRA_SPACE_RE = re.compile(r"\s+")

__all__ = [
    "TextCleaner",
    "TextCombiner",
    "FeatureEngineer",
    "clean_text",
]


def clean_text(
    text: Optional[str],
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_non_ascii: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """
    Clean a single text string with configurable options.

    Parameters
    ----------
    text : str | None
        Input text to normalize. Non-string values are treated as empty.
    lowercase : bool
        Convert to lowercase.
    remove_urls : bool
        Remove URL-like substrings.
    remove_emails : bool
        Remove email-like substrings.
    remove_non_ascii : bool
        Strip non-ASCII characters.
    collapse_whitespace : bool
        Replace runs of whitespace with a single space and strip ends.

    Returns
    -------
    str
        The normalized text.
    """
    if not isinstance(text, str):
        return ""

    s = text

    if lowercase:
        s = s.lower()
    if remove_urls:
        s = URL_RE.sub(" ", s)
    if remove_emails:
        s = EMAIL_RE.sub(" ", s)
    if remove_non_ascii:
        s = NON_ASCII_RE.sub(" ", s)
    if collapse_whitespace:
        s = EXTRA_SPACE_RE.sub(" ", s).strip()

    return s


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn Transformer for text cleaning/normalization.

    This transformer applies configurable text cleaning operations to each
    input text, ensuring consistent preprocessing between training and inference.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert text to lowercase.
    remove_urls : bool, default=True
        Remove URL patterns from text.
    remove_emails : bool, default=True
        Remove email patterns from text.
    remove_non_ascii : bool, default=True
        Remove non-ASCII characters.
    collapse_whitespace : bool, default=True
        Collapse multiple whitespace into single space.

    Examples
    --------
    >>> cleaner = TextCleaner()
    >>> cleaner.transform(["Check out https://example.com for more info!"])
    ['check out for more info!']
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_non_ascii: bool = True,
        collapse_whitespace: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_non_ascii = remove_non_ascii
        self.collapse_whitespace = collapse_whitespace

    def fit(self, X: Any, y: Any = None) -> "TextCleaner":
        return self

    def transform(self, X: Any) -> List[str]:
        """
        Transform input texts to cleaned versions.

        Parameters
        ----------
        X : array-like of str
            Input texts to clean.

        Returns
        -------
        list[str]
            Cleaned texts.
        """
        if isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, np.ndarray):
            texts = X.tolist()
        elif isinstance(X, (list, tuple)):
            texts = list(X)
        else:
            texts = [X] if isinstance(X, str) else list(X)

        return [
            clean_text(
                t,
                lowercase=self.lowercase,
                remove_urls=self.remove_urls,
                remove_emails=self.remove_emails,
                remove_non_ascii=self.remove_non_ascii,
                collapse_whitespace=self.collapse_whitespace,
            )
            for t in texts
        ]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "lowercase": self.lowercase,
            "remove_urls": self.remove_urls,
            "remove_emails": self.remove_emails,
            "remove_non_ascii": self.remove_non_ascii,
            "collapse_whitespace": self.collapse_whitespace,
        }

    def set_params(self, **params: Any) -> "TextCleaner":
        for key, value in params.items():
            setattr(self, key, value)
        return self


class TextCombiner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn Transformer for combining title and text columns.

    This transformer handles both DataFrame input (with separate title/text columns)
    and single string input (for inference). It ensures consistent feature
    construction across training and prediction.

    Parameters
    ----------
    title_col : str, default="title"
        Name of the title column in DataFrame.
    text_col : str, default="text"
        Name of the text column in DataFrame.
    separator : str, default=" "
        Separator to use when joining title and text.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit (for DataFrame input).

    Examples
    --------
    >>> combiner = TextCombiner(title_col="title", text_col="content")
    >>> df = pd.DataFrame({"title": ["Breaking News"], "content": ["Full article..."]})
    >>> combiner.fit_transform(df)
    ['Breaking News Full article...']
    """

    def __init__(
        self,
        *,
        title_col: str = "title",
        text_col: str = "text",
        separator: str = " ",
    ):
        self.title_col = title_col
        self.text_col = text_col
        self.separator = separator

    def fit(self, X: Any, y: Any = None) -> "TextCombiner":
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: Any) -> List[str]:
        """
        Transform input to combined text strings.

        Parameters
        ----------
        X : DataFrame, Series, array-like, or str
            Input data. Can be:
            - DataFrame with title_col and text_col columns
            - Series/list of strings (returned as-is after cleaning)
            - Single string (returned as single-element list)

        Returns
        -------
        list[str]
            Combined or cleaned text strings.
        """
        if isinstance(X, pd.DataFrame):
            return self._transform_dataframe(X)
        elif isinstance(X, pd.Series):
            return X.fillna("").tolist()
        elif isinstance(X, np.ndarray):
            return [str(x) if pd.notna(x) else "" for x in X]
        elif isinstance(X, (list, tuple)):
            return [str(x) if pd.notna(x) else "" for x in X]
        elif isinstance(X, str):
            return [X]
        else:
            return [str(X)]

    def _transform_dataframe(self, df: pd.DataFrame) -> List[str]:
        df_reset = df.reset_index(drop=True)
        title = (
            df_reset[self.title_col].fillna("")
            if self.title_col in df_reset.columns
            else pd.Series([""] * len(df_reset))
        )
        text = (
            df_reset[self.text_col].fillna("")
            if self.text_col in df_reset.columns
            else pd.Series([""] * len(df_reset))
        )
        combined = (title + self.separator + text).str.strip()
        return combined.tolist()

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "title_col": self.title_col,
            "text_col": self.text_col,
            "separator": self.separator,
        }

    def set_params(self, **params: Any) -> "TextCombiner":
        for key, value in params.items():
            setattr(self, key, value)
        return self


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Complete feature engineering pipeline for text classification.

    This transformer combines TextCombiner and TextCleaner into a single
    unified preprocessing step, ensuring consistent feature engineering
    across training and inference.

    The pipeline flow:
    1. Combine title + text (if DataFrame) or pass through (if string)
    2. Clean text (lowercase, remove URLs, etc.)

    Parameters
    ----------
    title_col : str, default="title"
        Name of the title column.
    text_col : str, default="text"
        Name of the text column.
    lowercase : bool, default=True
        Convert to lowercase.
    remove_urls : bool, default=True
        Remove URLs from text.
    remove_emails : bool, default=True
        Remove emails from text.
    remove_non_ascii : bool, default=True
        Remove non-ASCII characters.
    collapse_whitespace : bool, default=True
        Collapse multiple whitespace.

    Examples
    --------
    >>> fe = FeatureEngineer()
    >>> df = pd.DataFrame({"title": ["News"], "text": ["Content here..."]})
    >>> fe.fit_transform(df)
    ['news content here...']
    """

    def __init__(
        self,
        *,
        title_col: str = "title",
        text_col: str = "text",
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_non_ascii: bool = True,
        collapse_whitespace: bool = True,
    ):
        self.title_col = title_col
        self.text_col = text_col
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_non_ascii = remove_non_ascii
        self.collapse_whitespace = collapse_whitespace

        self._combiner = TextCombiner(
            title_col=title_col,
            text_col=text_col,
        )
        self._cleaner = TextCleaner(
            lowercase=lowercase,
            remove_urls=remove_urls,
            remove_emails=remove_emails,
            remove_non_ascii=remove_non_ascii,
            collapse_whitespace=collapse_whitespace,
        )

    def fit(self, X: Any, y: Any = None) -> "FeatureEngineer":
        self._combiner.fit(X, y)
        return self

    def transform(self, X: Any) -> List[str]:
        """
        Apply complete feature engineering pipeline.

        Parameters
        ----------
        X : DataFrame, Series, list, or str
            Input data.

        Returns
        -------
        list[str]
            Preprocessed text strings ready for vectorization.
        """
        combined = self._combiner.transform(X)
        cleaned = self._cleaner.transform(combined)
        return cleaned

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "title_col": self.title_col,
            "text_col": self.text_col,
            "lowercase": self.lowercase,
            "remove_urls": self.remove_urls,
            "remove_emails": self.remove_emails,
            "remove_non_ascii": self.remove_non_ascii,
            "collapse_whitespace": self.collapse_whitespace,
        }

    def set_params(self, **params: Any) -> "FeatureEngineer":
        for key, value in params.items():
            setattr(self, key, value)
        self._combiner.set_params(
            title_col=self.title_col,
            text_col=self.text_col,
        )
        self._cleaner.set_params(
            lowercase=self.lowercase,
            remove_urls=self.remove_urls,
            remove_emails=self.remove_emails,
            remove_non_ascii=self.remove_non_ascii,
            collapse_whitespace=self.collapse_whitespace,
        )
        return self


def create_preprocessing_pipeline(
    *,
    title_col: str = "title",
    text_col: str = "text",
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_non_ascii: bool = True,
    collapse_whitespace: bool = True,
) -> Pipeline:
    """
    Factory function to create a preprocessing pipeline.

    Returns
    -------
    Pipeline
        Scikit-learn Pipeline with FeatureEngineer as the first step.
    """
    return Pipeline(
        [
            (
                "feature_engineering",
                FeatureEngineer(
                    title_col=title_col,
                    text_col=text_col,
                    lowercase=lowercase,
                    remove_urls=remove_urls,
                    remove_emails=remove_emails,
                    remove_non_ascii=remove_non_ascii,
                    collapse_whitespace=collapse_whitespace,
                ),
            )
        ]
    )
