# Docker 部署指南

## 快速开始

### 方式一：Docker Compose 一键启动（推荐）

```bash
docker compose up --build
```

### 方式二：手动构建并运行

**构建镜像：**
```bash
docker build -t fake-news-detector:latest .
```

**运行容器：**
```bash
docker run --rm -p 8501:8501 fake-news-detector:latest
```

## 访问应用

启动成功后，在浏览器中访问：

```
http://localhost:8501
```

## 健康检查

应用启动后，可通过以下方式验证：

1. **浏览器访问**：打开 http://localhost:8501，应能看到 Streamlit 应用界面
2. **curl 命令**：
   ```bash
   curl http://localhost:8501
   ```
3. **查看容器日志**：
   ```bash
   docker logs fake-news-detector
   ```

## 停止服务

**Docker Compose 方式：**
```bash
docker compose down
```

**手动运行方式：**
按 `Ctrl+C` 停止容器（因使用了 `--rm` 参数，容器会自动删除）

## 镜像信息

- **基础镜像**：`python:3.11-slim`
- **运行用户**：非 root 用户 (`appuser`)
- **暴露端口**：`8501`
- **工作目录**：`/app`
