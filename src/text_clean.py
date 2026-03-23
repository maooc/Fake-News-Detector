#!/usr/bin/env python3
"""
Text cleaning utilities for the fake-news detector.

This module provides backward-compatible imports from the feature_engineering module.
All text cleaning logic is now centralized in feature_engineering.py for consistency.

For new code, prefer importing directly from feature_engineering:
    from feature_engineering import clean_text, TextCleaner
"""

from __future__ import annotations

from feature_engineering import clean_text, clean_many

__all__ = [
    "clean_text",
    "clean_many",
]
