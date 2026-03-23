#!/usr/bin/env python3
"""
Fake News Detector - Streamlit Web 应用

重构要点:
1. 完全依赖持久化的 Pipeline 进行预测，应用层不做任何文本处理
2. 训练和推理使用完全一致的特征预处理路径
3. 移除所有手动清洗、手动向量化的代码
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


def project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parents[1]


def get_default_pipeline_path() -> Path:
    """获取默认 Pipeline 路径"""
    return project_root() / "outputs" / "pipeline.joblib"


def load_pipeline(pipeline_path: Path) -> Any:
    """
    加载训练好的完整 Pipeline
    
    Parameters
    ----------
    pipeline_path : Path
        Pipeline 文件路径
        
    Returns
    -------
    Pipeline
        完整的特征工程+模型 Pipeline
        
    Raises
    ------
    FileNotFoundError
        当 Pipeline 文件不存在时
    """
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Pipeline not found: {pipeline_path}\n\n"
            "请确保已经训练模型：\n"
            "  python src/train_model.py --real data/True.csv --fake data/Fake.csv"
        )
    
    return joblib.load(pipeline_path)


def prepare_input(title: str, content: str) -> pd.DataFrame:
    """
    准备输入数据 - 仅做数据格式转换，不做任何清洗或特征工程
    
    所有文本处理（清洗、拼接、向量化）都由 Pipeline 内部完成
    
    Parameters
    ----------
    title : str
        标题文本（原始未处理）
    content : str
        正文文本（原始未处理）
        
    Returns
    -------
    pd.DataFrame
        原始输入数据，列名与训练时一致
    """
    # 直接传递原始文本，不做任何处理
    # Pipeline 中的 TextCombiner 和 TextCleaner 会处理后续步骤
    return pd.DataFrame([{
        "title": title,
        "text": content,
    }])


def predict(pipeline: Any, input_data: pd.DataFrame) -> tuple[str, float]:
    """
    执行预测 - 完全由 Pipeline 处理所有特征工程
    
    Parameters
    ----------
    pipeline : Pipeline
        完整的训练和推理 Pipeline
    input_data : pd.DataFrame
        原始输入数据（未经过任何处理）
        
    Returns
    -------
    tuple[str, float]
        (预测标签, 假新闻概率)
    """
    # Pipeline 内部自动完成: 文本拼接 → 清洗 → 向量化 → 预测
    prob_fake = float(pipeline.predict_proba(input_data)[0, 1])
    return prob_fake


def main():
    # 页面配置
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="centered"
    )
    
    st.title("📰 Fake News & Misinformation Detector")
    st.caption("基于 TF-IDF + RandomForest 的假新闻检测器")
    
    # 加载 Pipeline
    pipeline_path = get_default_pipeline_path()
    
    with st.sidebar:
        st.subheader("模型信息")
        st.code(f"Pipeline: {pipeline_path}")
        
        if pipeline_path.exists():
            st.success("✅ Pipeline 已加载")
        else:
            st.error("❌ Pipeline 未找到")
    
    try:
        pipeline = load_pipeline(pipeline_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    
    # 输入区域
    st.subheader("输入新闻内容")
    
    title = st.text_input("标题:", placeholder="请输入新闻标题...")
    content = st.text_area("正文:", height=200, placeholder="请输入新闻正文内容...")
    
    threshold = st.slider(
        "FAKE 判定阈值",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="概率高于此阈值则判定为假新闻"
    )
    
    # 预测按钮
    if st.button("🔍 分析", type="primary"):
        if not title.strip() and not content.strip():
            st.warning("请输入标题或正文内容")
            st.stop()
        
        # 准备输入数据 - 仅做格式转换，不做任何文本处理
        input_data = prepare_input(title, content)
        
        # 执行预测 - Pipeline 内部完成所有特征工程
        prob_fake = predict(pipeline, input_data)
        
        # 显示结果
        label = "FAKE" if prob_fake >= threshold else "REAL"
        confidence = prob_fake if label == "FAKE" else 1 - prob_fake
        
        st.divider()
        
        # 结果展示
        col1, col2 = st.columns(2)
        
        with col1:
            if label == "FAKE":
                st.error(f"### 预测结果: {label}")
            else:
                st.success(f"### 预测结果: {label}")
        
        with col2:
            st.metric("置信度", f"{confidence:.1%}")
        
        # 概率进度条
        st.write("**假新闻概率:**")
        st.progress(prob_fake, text=f"{prob_fake:.1%}")
        
        # 详细信息
        with st.expander("查看详细信息"):
            st.json({
                "label": label,
                "fake_probability": round(prob_fake, 4),
                "real_probability": round(1 - prob_fake, 4),
                "confidence": round(confidence, 4),
                "threshold": threshold,
            })


if __name__ == "__main__":
    main()
