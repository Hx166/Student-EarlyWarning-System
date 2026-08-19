# Docker 部署

## 1. 镜像构建

项目根目录提供 `Dockerfile`，基于 `python:3.11-slim`：

```bash
docker build -t student-early-warning .
```

构建时安装 `requirements.txt` 全部依赖，并安装 OpenCV/MediaPipe 运行所需的系统库（`libgl1`、`libglib2.0-0`、`libgomp1`）。

## 2. 一键编排

`docker-compose.yml` 编排 App + MySQL + InfluxDB（后两者可选）：

```bash
docker-compose up -d --build
```

服务：

| 服务 | 端口 | 说明 |
|------|------|------|
| app | 5000 | Web 服务与 Dashboard |
| mysql | 3306 | 关系库（可选） |
| influxdb | 8086 | 时序库（可选） |

App 默认使用 SQLite（`data/` 卷挂载到 `/app/data`）。
切换 MySQL 需修改 `configs/database.yaml`：

```yaml
database:
  backend: "mysql"
mysql:
  host: "mysql"
  port: 3306
  user: "root"
  password: "root"
  database: "early_warning"
```

## 3. 单独运行

```bash
docker run --rm -p 5000:5000 -v "$(pwd)/data:/app/data" student-early-warning
```

访问 `http://localhost:5000` 打开 Dashboard。

## 4. 注意事项

- ECharts 已本地化（`src/frontend/static/js/echarts.min.js`），离线/内网可用，不依赖 CDN。
- `torch`、`shap`、`mediapipe` 依赖较大，首次构建耗时较长，建议使用国内镜像源或缓存层。
