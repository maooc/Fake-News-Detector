FROM python:3.11-slim

# 安装 scikit-learn / matplotlib 等库的运行依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libfreetype6 \
    libpng16-16 \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 设置工作目录
WORKDIR /app

# 先拷贝依赖文件以利用层缓存
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码（保持 src/ 和 outputs/ 在同一目录层级）
COPY src/ ./src/
COPY outputs/ ./outputs/

# 更改文件所有权给非 root 用户
RUN chown -R appuser:appuser /app

# 切换到非 root 用户
USER appuser

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 启动 Streamlit 应用（绑定 0.0.0.0 以允许外部访问）
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
