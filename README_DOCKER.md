# Fake News Detector - Docker 部署指南

本文档说明如何使用 Docker 部署 Fake News Detector 应用。

## 前置要求

- Docker Engine 20.10+
- Docker Compose v2.0+（可选）

## 快速开始

### 方式一：使用 Docker Compose（推荐）

```bash
# 从 Fake-News-Detector 目录执行
docker compose up --build
```

### 方式二：手动构建并运行

```bash
# 构建镜像
docker build -t fake-news-detector:latest .

# 运行容器
docker run --rm -p 8501:8501 fake-news-detector:latest
```

### 方式三：后台运行

```bash
# 使用 docker compose 后台运行
docker compose up -d --build

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

## 访问应用

应用启动后，在浏览器中访问：

```
http://localhost:8501
```

## 健康检查

### 方式一：浏览器访问

直接访问 http://localhost:8501，看到 Streamlit 界面即表示服务正常。

### 方式二：命令行检查

```bash
# 检查健康端点
curl http://localhost:8501/_stcore/health

# 或使用 wget
wget -q -O - http://localhost:8501/_stcore/health
```

### 方式三：查看容器状态

```bash
# 查看容器健康状态
docker ps --format "table {{.Names}}\t{{.Status}}"

# 或使用 docker compose
docker compose ps
```

## 常用命令

```bash
# 重新构建镜像
docker compose build --no-cache

# 查看容器日志
docker compose logs -f

# 进入容器调试
docker compose exec fake-news-detector /bin/bash

# 停止并删除容器
docker compose down

# 停止并删除容器及镜像
docker compose down --rmi local
```

## 目录结构说明

```
Fake-News-Detector/
├── Dockerfile              # Docker 镜像构建文件
├── docker-compose.yml      # Docker Compose 配置
├── .dockerignore           # Docker 构建忽略文件
├── requirements.txt        # Python 依赖
├── src/                    # 源代码目录
│   └── streamlit_app.py    # Streamlit 应用入口
└── outputs/                # 模型文件目录（已包含在镜像中）
    ├── model.joblib
    ├── vectorizer.joblib
    └── pipeline.joblib
```

## 注意事项

1. **端口冲突**：如果宿主机 8501 端口已被占用，可修改 `docker-compose.yml` 中的端口映射，例如改为 `"8502:8501"`。

2. **模型文件**：模型文件已打包在镜像中，无需额外挂载。

3. **非 root 用户**：容器内使用 `appuser` 非 root 用户运行，提高安全性。

4. **镜像大小优化**：
   - 使用 `python:3.11-slim` 基础镜像
   - 通过 `.dockerignore` 排除不必要文件
   - 清理 apt 缓存

## 故障排查

### 容器无法启动

```bash
# 查看详细日志
docker compose logs fake-news-detector

# 检查镜像是否构建成功
docker images | grep fake-news-detector
```

### 无法访问应用

1. 确认容器正在运行：`docker ps`
2. 确认端口映射正确：`docker port fake-news-detector`
3. 检查防火墙设置

### 模型加载失败

确认 `outputs/` 目录下存在以下文件：
- model.joblib
- vectorizer.joblib
- pipeline.joblib
