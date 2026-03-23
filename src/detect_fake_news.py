#!/usr/bin/env python3
"""
Fake News Detector 推理脚本

重构要点:
1. 完全依赖持久化的 Pipeline 进行预测，Pipeline 外不做任何文本处理
2. 训练和推理使用完全一致的特征预处理路径
3. 移除所有手动特征构建、手动清洗、手动向量化的代码
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_pipeline(pipeline_path: str | None = None) -> Any:
    """
    加载训练好的完整 Pipeline
    
    Parameters
    ----------
    pipeline_path : str, optional
        Pipeline 文件路径，默认为 outputs/pipeline.joblib
        
    Returns
    -------
    Pipeline
        完整的特征工程+模型 Pipeline
        
    Raises
    ------
    FileNotFoundError
        当 Pipeline 文件不存在时
    """
    if pipeline_path:
        path = Path(pipeline_path)
    else:
        path = Path("outputs/pipeline.joblib")
    
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline not found: {path}\n"
            "Please train the model first using train_model.py"
        )
    
    return joblib.load(path)


def prepare_input(
    text: str | None = None,
    title: str | None = None,
    content: str | None = None,
) -> pd.DataFrame:
    """
    准备输入数据 - 仅做数据格式转换，不做任何清洗或特征工程
    
    所有文本处理（清洗、拼接、向量化）都由 Pipeline 内部完成
    
    Parameters
    ----------
    text : str, optional
        完整文本（title + content 已拼接）
    title : str, optional
        标题文本
    content : str, optional
        正文文本
        
    Returns
    -------
    pd.DataFrame
        原始输入数据，列名与训练时一致
    """
    if text is not None:
        # 纯文本输入 - 放入 text 列，title 为空
        # Pipeline 中的 TextCombiner 会处理拼接
        return pd.DataFrame([{
            "title": "",
            "text": text,
        }])
    elif title is not None or content is not None:
        # 结构化输入 - 直接传递原始文本，不做任何处理
        return pd.DataFrame([{
            "title": title or "",
            "text": content or "",
        }])
    else:
        raise ValueError("Must provide either --text OR (--title and/or --content)")


def predict(pipeline: Any, input_data: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    """
    执行预测 - 完全由 Pipeline 处理所有特征工程
    
    Parameters
    ----------
    pipeline : Pipeline
        完整的训练和推理 Pipeline
    input_data : pd.DataFrame
        原始输入数据（未经过任何处理）
    threshold : float, default=0.5
        分类阈值
        
    Returns
    -------
    dict
        预测结果
    """
    # Pipeline 内部自动完成: 文本拼接 → 清洗 → 向量化 → 预测
    prob = float(pipeline.predict_proba(input_data)[0, 1])
    
    label = "FAKE" if prob >= threshold else "REAL"
    confidence = prob if label == "FAKE" else 1 - prob
    
    return {
        "label": label,
        "fake_probability": prob,
        "real_probability": 1 - prob,
        "confidence": confidence,
        "threshold": threshold,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Detect fake news using trained pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用结构化输入（推荐）
  python detect_fake_news.py --title "Breaking News" --content "This is the article text..."
  
  # 使用预拼接文本
  python detect_fake_news.py --text "Breaking News This is the article text..."
  
  # 指定 Pipeline 路径
  python detect_fake_news.py --pipeline outputs/pipeline.joblib --title "..." --content "..."
        """
    )
    
    ap.add_argument("--pipeline", help="Path to pipeline.joblib (default: outputs/pipeline.joblib)")
    
    # 输入参数
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Full article text (title + content combined)")
    input_group.add_argument("--title", help="Article title")
    
    ap.add_argument("--content", help="Article content (use with --title)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for FAKE (default: 0.50)")
    ap.add_argument("--json", action="store_true",
                    help="Output results as JSON")
    
    args = ap.parse_args()
    
    # 验证参数
    if args.content and not args.title:
        ap.error("--content must be used with --title")
    
    # 加载 Pipeline
    try:
        pipeline = load_pipeline(args.pipeline)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 准备输入数据 - 仅做格式转换，不做任何文本处理
    input_data = prepare_input(
        text=args.text,
        title=args.title,
        content=args.content,
    )
    
    # 执行预测 - Pipeline 内部完成所有特征工程
    result = predict(pipeline, input_data, threshold=args.threshold)
    
    # 输出结果
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"Prediction Result")
        print(f"{'='*50}")
        print(f"Label:           {result['label']}")
        print(f"Confidence:      {result['confidence']:.1%}")
        print(f"Fake Prob:       {result['fake_probability']:.3f}")
        print(f"Real Prob:       {result['real_probability']:.3f}")
        print(f"Threshold:       {result['threshold']:.2f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
