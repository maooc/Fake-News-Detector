# Docker 部署指南

## 项目概述

假新闻检测器应用的 Docker 容器化部署方案。

## 先决条件

确保已安装以下软件：
- Docker
- Docker Compose

## 快速开始

### 方式一：使用 Docker Compose（推荐）

一键构建并启动服务：

```bash
docker compose up --build
```

后台运行：

```bash
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f
```

停止服务：

```bash
docker compose down
```

### 方式二：使用原生 Docker 命令

构建镜像：

```bash
docker build -t fake-news-detector:latest .
```

运行容器：

```bash
docker run --rm -p 8501:8501 fake-news-detector:latest
```

后台运行：

```bash
docker run -d --name fake-news-detector -p 8501:8501 --restart unless-stopped fake-news-detector:latest
```

查看运行状态：

```bash
docker ps
```

查看容器日志：

```bash
docker logs -f fake-news-detector
```

停止容器：

```bash
docker stop fake-news-detector
```

## 访问应用

服务启动后，在浏览器中访问：

```
http://localhost:8501
```

## 健康检查

Docker 容器内置健康检查机制，可通过以下命令查看健康状态：

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

或使用 Docker Compose：

```bash
docker compose ps
```

健康状态为 `healthy` 表示服务正常运行。

## 端口说明

- 容器内部端口：8501
- 宿主机映射端口：8501

如需修改宿主机端口，请修改 `docker-compose.yml` 中的端口映射配置。

## 安全说明

- 容器内使用非 root 用户（appuser）运行应用，提高安全性
- 所有依赖包在构建时安装，确保环境一致性
- 不包含不必要的系统组件，减小镜像攻击面
