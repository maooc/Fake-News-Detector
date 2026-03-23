#!/usr/bin/env python3
"""
Fake News Detector 训练脚本

重构要点:
1. 使用统一的 Pipeline 封装所有特征工程步骤
2. 训练和推理使用完全一致的数据预处理流程
3. 支持灵活的文本字段组合和清洗配置
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion

# 导入自定义特征工程模块
from features import TextCleaner, TextCombiner, TextStatsExtractor

# -----------------------------
# 常量定义
# -----------------------------
LABELS: Final[Tuple[str, str]] = ("REAL", "FAKE")

DEFAULT_TEXT_COLUMNS = ["title", "text"]


# -----------------------------
# 工具函数
# -----------------------------
def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_any(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """读取 CSV，自动处理编码"""
    try:
        return pd.read_csv(path, nrows=nrows, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, encoding="latin-1")


def load_and_prepare_data(
    real_path: Path,
    fake_path: Path,
    text_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    加载并准备训练数据
    
    返回包含原始字段的 DataFrame，特征工程由 Pipeline 处理
    """
    text_columns = text_columns or DEFAULT_TEXT_COLUMNS
    
    df_real = read_csv_any(real_path)
    df_fake = read_csv_any(fake_path)
    
    # 添加标签
    df_real["label"] = 0  # REAL
    df_fake["label"] = 1  # FAKE
    
    # 合并数据
    df = pd.concat([df_real, df_fake], ignore_index=True)
    
    # 确保所需列存在
    for col in text_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in data. Available columns: {list(df.columns)}")
    
    return df


def create_feature_pipeline(
    text_columns: list[str] | None = None,
    use_text_stats: bool = False,
) -> Pipeline:
    """
    创建特征工程 Pipeline
    
    Pipeline 结构:
    1. TextCombiner: 拼接多个文本字段 (如 title + text)
    2. TextCleaner: 清洗文本 (小写、去 URL、去邮箱等)
    3. (可选) TextStatsExtractor: 提取文本统计特征
    4. TfidfVectorizer: TF-IDF 特征提取
    
    Parameters
    ----------
    text_columns : list of str, optional
        要拼接的文本字段，默认 ["title", "text"]
    use_text_stats : bool, default=False
        是否使用文本统计特征
        
    Returns
    -------
    Pipeline
        特征工程 Pipeline
    """
    text_columns = text_columns or DEFAULT_TEXT_COLUMNS
    
    # 基础文本处理步骤
    text_processing_steps = [
        ("combiner", TextCombiner(columns=text_columns, separator=" ", handle_missing="fill")),
        ("cleaner", TextCleaner(
            lowercase=True,
            remove_urls=True,
            remove_emails=True,
            remove_non_ascii=True,
            collapse_whitespace=True,
        )),
    ]
    
    if use_text_stats:
        # 使用 FeatureUnion 合并 TF-IDF 和统计特征
        feature_union = FeatureUnion([
            ("tfidf", TfidfVectorizer(
                sublinear_tf=True,
                stop_words="english",
                ngram_range=(1, 3),
                max_df=0.8,
                min_df=3,
                max_features=20_000,
            )),
            ("stats", TextStatsExtractor(features=["char_count", "word_count", "exclamation_count"])),
        ])
        text_processing_steps.append(("features", feature_union))
    else:
        # 仅使用 TF-IDF
        text_processing_steps.append(("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=3,
            max_features=20_000,
        )))
    
    return Pipeline(text_processing_steps)


def create_full_pipeline(
    text_columns: list[str] | None = None,
    use_text_stats: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """
    创建完整的训练和推理 Pipeline
    
    该 Pipeline 包含从原始数据到预测结果的完整流程，
    确保训练和推理使用完全一致的特征工程。
    
    Parameters
    ----------
    text_columns : list of str, optional
        要拼接的文本字段
    use_text_stats : bool, default=False
        是否使用文本统计特征
    random_state : int, default=42
        随机种子
        
    Returns
    -------
    Pipeline
        完整 Pipeline，包含特征工程和分类器
    """
    feature_pipe = create_feature_pipeline(text_columns, use_text_stats)
    
    return Pipeline([
        ("features", feature_pipe),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])


# -----------------------------
# 可视化工具
# -----------------------------
def plot_confusion_matrix(cm: np.ndarray, out: Path, title: str = "Confusion Matrix") -> None:
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LABELS)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(LABELS)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_curve(
    x: np.ndarray,
    y: np.ndarray,
    out: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """绘制曲线"""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y, linewidth=2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# -----------------------------
# 主训练流程
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train Fake News Detector with unified feature engineering pipeline"
    )
    ap.add_argument("--real", required=True, help="Path to True.csv")
    ap.add_argument("--fake", required=True, help="Path to Fake.csv")
    ap.add_argument(
        "--text-cols",
        nargs="+",
        default=["title", "text"],
        help="Text columns to combine. Default: title text",
    )
    ap.add_argument("--use-text-stats", action="store_true", help="Include text statistics features")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test set ratio (default: 0.2)")
    ap.add_argument("--cv-splits", type=int, default=5, help="Cross-validation folds (default: 5)")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")
    a = ap.parse_args()

    outdir = ensure_dir(Path(a.outdir))
    charts = ensure_dir(outdir / "charts")

    # 1) 加载数据
    print(f"Loading data from {a.real} and {a.fake}...")
    df = load_and_prepare_data(
        Path(a.real),
        Path(a.fake),
        text_columns=a.text_cols,
    )
    print(f"Loaded {len(df)} samples ({(df['label']==0).sum()} REAL, {(df['label']==1).sum()} FAKE)")

    # 2) 准备 X, y
    # X 保持为 DataFrame，让 Pipeline 处理特征工程
    X = df
    y = df["label"].values

    # 3) 划分训练/测试集
    from sklearn.model_selection import train_test_split
    
    # 分层抽样保持类别平衡
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=a.test_size,
        random_state=a.random_state,
        stratify=y,
    )
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 4) 创建完整 Pipeline
    print("Creating pipeline...")
    pipeline = create_full_pipeline(
        text_columns=a.text_cols,
        use_text_stats=a.use_text_stats,
        random_state=a.random_state,
    )

    # 5) 交叉验证
    print(f"Running {a.cv_splits}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=a.cv_splits, shuffle=True, random_state=a.random_state)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"CV F1 (macro): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    # 6) 训练模型
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 7) 评估
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "cv_f1_macro_mean": float(cv_f1.mean()),
        "cv_f1_macro_std": float(cv_f1.std()),
        "report": classification_report(y_test, y_pred, target_names=LABELS, output_dict=True),
    }

    print(f"\nTest Accuracy: {metrics['accuracy']:.3f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    # 8) 保存指标和图表
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, charts / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plot_curve(fpr, tpr, charts / "roc_curve.png", "ROC Curve", "FPR", "TPR")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plot_curve(rec, prec, charts / "pr_curve.png", "Precision-Recall Curve", "Recall", "Precision")

    # 9) 保存模型
    # 保存完整 Pipeline (推荐，包含所有特征工程)
    pipeline_path = outdir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"\nSaved complete pipeline to: {pipeline_path}")
    
    # 同时保存单独组件以兼容旧代码
    feature_pipe = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    joblib.dump(feature_pipe, outdir / "vectorizer.joblib")
    joblib.dump(clf, outdir / "model.joblib")
    
    # 保存 Pipeline 配置信息
    config = {
        "text_columns": a.text_cols,
        "use_text_stats": a.use_text_stats,
        "random_state": a.random_state,
    }
    (outdir / "pipeline_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
