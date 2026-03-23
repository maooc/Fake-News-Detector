#!/usr/bin/env python3
"""
特征工程模块 - 提供可复用、可扩展的特征提取 Transformer

该模块包含:
- TextCleaner: 文本清洗 Transformer
- TextCombiner: 多字段文本拼接 Transformer
- TextStatsExtractor: 文本统计特征提取器 (可扩展)
"""

from __future__ import annotations

from .transformers import TextCleaner, TextCombiner, TextStatsExtractor

__all__ = [
    "TextCleaner",
    "TextCombiner", 
    "TextStatsExtractor",
]
