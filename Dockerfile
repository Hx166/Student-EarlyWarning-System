# Student-EarlyWarning-System 多阶段构建 Dockerfile
# 基础镜像：Python 3.11

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OpenCV / MediaPipe 运行所需的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单以利用构建缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码与前端静态资源
COPY . .

EXPOSE 5000

CMD ["python", "src/main.py"]
