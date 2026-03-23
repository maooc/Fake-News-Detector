#!/usr/bin/env python3
"""
Feature extraction transformers for fake news detection.
Provides scikit-learn compatible transformers for text preprocessing and feature engineering.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from text_clean import clean_text

__all__ = [
    "TextCleaner",
    "TextCombiner",
    "ItemSelector",
    "TextStatsExtractor",
]


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer for text normalization/cleaning.
    
    Wraps the clean_text function into a transformer that can be
    integrated into a scikit-learn Pipeline.
    
    Parameters
    ----------
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
    
    Examples
    --------
    >>> cleaner = TextCleaner()
    >>> cleaned = cleaner.transform(["Check https://example.com for info!"])
    >>> cleaned[0]
    'check for info!'
    """

    def __init__(
        self,
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
        """Fit the transformer (no-op for stateless transformer)."""
        return self

    def transform(self, X: Iterable[Optional[str]]) -> List[str]:
        """
        Clean a sequence of text strings.
        
        Parameters
        ----------
        X : iterable of str | None
            Sequence of raw texts to clean.
            
        Returns
        -------
        list of str
            Cleaned text strings.
        """
        return [
            clean_text(
                text,
                lowercase=self.lowercase,
                remove_urls=self.remove_urls,
                remove_emails=self.remove_emails,
                remove_non_ascii=self.remove_non_ascii,
                collapse_whitespace=self.collapse_whitespace,
            )
            for text in X
        ]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        return np.array(["cleaned_text"])


class TextCombiner(BaseEstimator, TransformerMixin):
    """
    A transformer that combines title and text columns into a single string.
    
    Useful for datasets that have separate title and article text fields.
    
    Parameters
    ----------
    title_col : str
        Name of the title column/key.
    text_col : str
        Name of the main text column/key.
    separator : str
        String to use between title and text.
    
    Examples
    --------
    >>> combiner = TextCombiner('title', 'body')
    >>> df = pd.DataFrame({'title': ['Hello'], 'body': ['World']})
    >>> result = combiner.transform(df)
    >>> result[0]
    'Hello World'
    """

    def __init__(
        self,
        title_col: str = "title",
        text_col: str = "text",
        separator: str = " ",
    ):
        self.title_col = title_col
        self.text_col = text_col
        self.separator = separator

    def fit(self, X: Any, y: Any = None) -> "TextCombiner":
        """Fit the transformer (no-op for stateless transformer)."""
        return self

    def transform(self, X: Union[pd.DataFrame, Sequence[Dict[str, Any]]]) -> List[str]:
        """
        Combine title and text fields from input data.
        
        Parameters
        ----------
        X : DataFrame or sequence of dicts
            Input data containing title and text fields.
            
        Returns
        -------
        list of str
            Combined text strings.
        """
        if isinstance(X, pd.DataFrame):
            titles = X.get(self.title_col, X.get("title", pd.Series([""] * len(X)))).fillna("")
            texts = X.get(self.text_col, X.get("text", pd.Series([""] * len(X)))).fillna("")
            combined = (titles + self.separator + texts).str.strip()
            return combined.tolist()
        else:
            result = []
            for item in X:
                if isinstance(item, dict):
                    title = item.get(self.title_col, item.get("title", "")) or ""
                    text = item.get(self.text_col, item.get("text", "")) or ""
                    combined = f"{title}{self.separator}{text}".strip()
                    result.append(combined)
                else:
                    result.append(str(item))
            return result

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        return np.array(["combined_text"])


class ItemSelector(BaseEstimator, TransformerMixin):
    """
    A transformer that selects a subset of columns from input data.
    
    Useful for Pipeline when you need to route specific columns to different
    feature extractors.
    
    Parameters
    ----------
    key : str or list of str
        Column name(s) to select.
    
    Examples
    --------
    >>> selector = ItemSelector('text')
    >>> df = pd.DataFrame({'text': ['hello'], 'other': ['world']})
    >>> selector.transform(df)
    ['hello']
    """

    def __init__(self, key: Union[str, List[str]]):
        self.key = key

    def fit(self, X: Any, y: Any = None) -> "ItemSelector":
        """Fit the transformer (no-op for stateless transformer)."""
        return self

    def transform(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> Any:
        """
        Select specific column(s) from input data.
        
        Parameters
        ----------
        X : DataFrame or dict
            Input data.
            
        Returns
        -------
        Series, array, or scalar
            Selected data.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.key]
        elif isinstance(X, dict):
            return X[self.key]
        else:
            return X


class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """
    A transformer that extracts statistical features from text.
    
    Extracts features like:
    - Text length
    - Number of words
    - Average word length
    - Number of uppercase words
    - Number of exclamation marks
    
    Parameters
    ----------
    normalize : bool
        If True, normalize features by text length.
    
    Examples
    --------
    >>> extractor = TextStatsExtractor()
    >>> features = extractor.transform(["Hello World! This is a TEST."])
    >>> features.shape
    (1, 5)
    """

    def __init__(self, normalize: bool = False):
        self.normalize = normalize

    def fit(self, X: Any, y: Any = None) -> "TextStatsExtractor":
        """Fit the transformer (no-op for stateless transformer)."""
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        """
        Extract statistical features from text.
        
        Parameters
        ----------
        X : iterable of str
            Sequence of texts to extract features from.
            
        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Extracted features.
        """
        features = []
        for text in X:
            text = str(text)
            # Basic stats
            char_count = len(text)
            word_count = len(text.split())
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Special patterns
            upper_word_count = sum(1 for word in words if word.isupper() and len(word) > 1)
            exclamation_count = text.count("!")
            question_count = text.count("?")
            
            # Optionally normalize
            if self.normalize and char_count > 0:
                upper_word_count /= char_count
                exclamation_count /= char_count
                question_count /= char_count
            
            features.append([
                char_count,
                word_count,
                avg_word_length,
                upper_word_count,
                exclamation_count,
                question_count,
            ])
        
        return np.array(features, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        return np.array([
            "char_count",
            "word_count",
            "avg_word_length",
            "upper_word_count",
            "exclamation_count",
            "question_count",
        ])
