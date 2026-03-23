#!/usr/bin/env python3
"""
自定义 Scikit-learn Transformer 用于特征工程

所有 Transformer 都继承自 BaseEstimator 和 TransformerMixin，
确保与 Scikit-learn Pipeline 完全兼容。
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    文本清洗 Transformer
    
    将 text_clean.py 中的清洗逻辑封装为 Scikit-learn Transformer，
    使其可以无缝集成到 Pipeline 中。
    
    Parameters
    ----------
    lowercase : bool, default=True
        是否转换为小写
    remove_urls : bool, default=True
        是否移除 URL
    remove_emails : bool, default=True
        是否移除邮箱
    remove_non_ascii : bool, default=True
        是否移除非 ASCII 字符
    collapse_whitespace : bool, default=True
        是否合并多余空白
    """
    
    # 预编译正则表达式 (类级别，只编译一次)
    URL_RE = re.compile(r"https?://\S+")
    EMAIL_RE = re.compile(r"\S+@\S+")
    NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
    EXTRA_SPACE_RE = re.compile(r"\s+")
    
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
    
    def fit(self, X: Sequence[str], y: Optional[Sequence] = None) -> "TextCleaner":
        """无状态 transformer，直接返回 self"""
        return self
    
    def transform(self, X: Sequence[str]) -> np.ndarray:
        """
        清洗文本序列
        
        Parameters
        ----------
        X : sequence of str
            输入文本序列
            
        Returns
        -------
        np.ndarray
            清洗后的文本数组，shape=(n_samples,)
        """
        cleaned = [self._clean_text(str(text) if text is not None else "") for text in X]
        return np.array(cleaned)
    
    def _clean_text(self, text: str) -> str:
        """清洗单个文本"""
        s = text
        
        if self.lowercase:
            s = s.lower()
        if self.remove_urls:
            s = self.URL_RE.sub(" ", s)
        if self.remove_emails:
            s = self.EMAIL_RE.sub(" ", s)
        if self.remove_non_ascii:
            s = self.NON_ASCII_RE.sub(" ", s)
        if self.collapse_whitespace:
            s = self.EXTRA_SPACE_RE.sub(" ", s).strip()
        
        return s


class TextCombiner(BaseEstimator, TransformerMixin):
    """
    文本字段拼接 Transformer
    
    用于将多个文本字段（如 title + text）拼接成一个字段，
    支持 DataFrame 输入和字典列表输入。
    
    Parameters
    ----------
    columns : list of str
        要拼接的列名列表，按顺序拼接
    separator : str, default=" "
        字段之间的分隔符
    handle_missing : str, default="fill"
        处理缺失值的方式: "fill" 填充空字符串, "drop" 丢弃, "error" 报错
    """
    
    def __init__(
        self,
        columns: List[str],
        separator: str = " ",
        handle_missing: str = "fill",
    ):
        self.columns = columns
        self.separator = separator
        self.handle_missing = handle_missing
        
        if handle_missing not in ("fill", "drop", "error"):
            raise ValueError(f"handle_missing must be 'fill', 'drop', or 'error', got {handle_missing}")
    
    def fit(self, X, y: Optional[Sequence] = None) -> "TextCombiner":
        """验证输入数据格式"""
        if isinstance(X, pd.DataFrame):
            missing_cols = set(self.columns) - set(X.columns)
            if missing_cols and self.handle_missing == "error":
                raise ValueError(f"Missing columns: {missing_cols}")
        return self
    
    def transform(self, X) -> np.ndarray:
        """
        拼接文本字段
        
        Parameters
        ----------
        X : pd.DataFrame or list of dict
            输入数据，包含需要拼接的列
            
        Returns
        -------
        np.ndarray
            拼接后的文本数组，shape=(n_samples,)
        """
        if isinstance(X, pd.DataFrame):
            return self._transform_dataframe(X)
        elif isinstance(X, (list, np.ndarray)):
            return self._transform_records(X)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")
    
    def _transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """处理 DataFrame 输入"""
        parts = []
        for col in self.columns:
            if col in df.columns:
                parts.append(df[col].fillna("").astype(str))
            elif self.handle_missing == "error":
                raise ValueError(f"Column '{col}' not found in DataFrame")
            else:
                # fill with empty strings
                parts.append(pd.Series([""] * len(df), index=df.index))
        
        combined = parts[0]
        for part in parts[1:]:
            combined = combined + self.separator + part
        
        return combined.str.strip().values
    
    def _transform_records(self, records: Sequence) -> np.ndarray:
        """处理字典列表输入"""
        results = []
        for record in records:
            if isinstance(record, dict):
                parts = []
                for col in self.columns:
                    val = record.get(col, "")
                    if val is None:
                        val = ""
                    parts.append(str(val))
                combined = self.separator.join(parts).strip()
                results.append(combined)
            else:
                # 如果是字符串，直接返回
                results.append(str(record))
        return np.array(results)


class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """
    文本统计特征提取器
    
    提取文本的统计特征，如长度、词数、平均词长等。
    这些特征可以作为 TF-IDF 的补充，提供给模型更多信号。
    
    Parameters
    ----------
    features : list of str, default=None
        要提取的特征列表。如果为 None，提取所有特征。
        可选特征: ["char_count", "word_count", "avg_word_length", 
                  "exclamation_count", "question_count", "uppercase_ratio"]
    """
    
    AVAILABLE_FEATURES = [
        "char_count",
        "word_count", 
        "avg_word_length",
        "exclamation_count",
        "question_count",
        "uppercase_ratio",
    ]
    
    def __init__(self, features: Optional[List[str]] = None):
        self.features = features or self.AVAILABLE_FEATURES
        
        # 验证特征名称
        invalid = set(self.features) - set(self.AVAILABLE_FEATURES)
        if invalid:
            raise ValueError(f"Unknown features: {invalid}. Available: {self.AVAILABLE_FEATURES}")
    
    def fit(self, X: Sequence[str], y: Optional[Sequence] = None) -> "TextStatsExtractor":
        """无状态 transformer，直接返回 self"""
        return self
    
    def transform(self, X: Sequence[str]) -> np.ndarray:
        """
        提取文本统计特征
        
        Parameters
        ----------
        X : sequence of str
            输入文本序列
            
        Returns
        -------
        np.ndarray
            特征矩阵，shape=(n_samples, n_features)
        """
        X = np.array([str(x) if x is not None else "" for x in X])
        
        feature_matrix = []
        for feat_name in self.features:
            extractor = getattr(self, f"_extract_{feat_name}")
            feature_matrix.append(extractor(X))
        
        return np.column_stack(feature_matrix)
    
    def _extract_char_count(self, X: np.ndarray) -> np.ndarray:
        """字符数"""
        return np.array([len(text) for text in X]).reshape(-1, 1)
    
    def _extract_word_count(self, X: np.ndarray) -> np.ndarray:
        """词数"""
        return np.array([len(text.split()) for text in X]).reshape(-1, 1)
    
    def _extract_avg_word_length(self, X: np.ndarray) -> np.ndarray:
        """平均词长"""
        result = []
        for text in X:
            words = text.split()
            if words:
                result.append(sum(len(w) for w in words) / len(words))
            else:
                result.append(0.0)
        return np.array(result).reshape(-1, 1)
    
    def _extract_exclamation_count(self, X: np.ndarray) -> np.ndarray:
        """感叹号数量"""
        return np.array([text.count("!") for text in X]).reshape(-1, 1)
    
    def _extract_question_count(self, X: np.ndarray) -> np.ndarray:
        """问号数量"""
        return np.array([text.count("?") for text in X]).reshape(-1, 1)
    
    def _extract_uppercase_ratio(self, X: np.ndarray) -> np.ndarray:
        """大写字母比例"""
        result = []
        for text in X:
            if len(text) > 0:
                result.append(sum(1 for c in text if c.isupper()) / len(text))
            else:
                result.append(0.0)
        return np.array(result).reshape(-1, 1)
    
    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """返回特征名称"""
        return np.array(self.features)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    恒等 Transformer
    
    用于 Pipeline 中需要占位或透传数据的场景
    """
    
    def fit(self, X, y: Optional[Sequence] = None) -> "IdentityTransformer":
        return self
    
    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.array(X)
