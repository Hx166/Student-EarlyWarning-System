# 多模态视觉驱动的学生学业状态分析与预警系统

> SRTP 科研项目（Student Research Training Program）

基于 **OpenCV + MediaPipe + XGBoost + Flask** 的学生课堂行为分析与学业状态预警系统。

---

# 项目简介

本项目旨在构建一套基于计算机视觉（Computer Vision）的学生课堂状态分析平台，通过摄像头实时采集学生课堂行为信息，结合机器学习算法，对学生课堂参与度、疲劳状态及学习风险进行分析，为教师提供客观、实时、可解释的辅助决策依据。

系统采用模块化设计，将视觉特征提取、课堂行为分析、数据存储、可视化展示以及机器学习预测进行解耦，实现良好的可扩展性与可维护性。

---

# 项目目标

当前项目采用迭代开发模式。

## V0.1（当前目标）

完成一个能够运行、演示的课堂行为分析系统，实现：

* 摄像头实时采集
* 人脸检测
* MediaPipe 468 关键点检测
* 眼部 EAR 闭眼检测
* Head Pose 姿态估计
* 疲劳状态分析
* 课堂参与度分析
* SQLite 数据存储
* Flask Dashboard 可视化

---

## V0.2

增加：

* 行为事件统计
* SQLite 数据分析
* Dashboard 优化
* Feature Engineering

---

## V0.3

增加：

* XGBoost 学业风险预测
* 风险评分
* 模型训练
* 模型评估

---

## V0.4

增加：

* Flask 后台管理
* 实时课堂看板
* SHAP 可解释性分析

---

## V1.0

完成：

* LSTM 时间序列预测
* 多模态数据融合
* MySQL + InfluxDB
* 完整部署方案

---

# 技术路线

```text
Camera
    │
    ▼
OpenCV
    │
    ▼
MediaPipe
    │
    ▼
Face Detection
    │
    ▼
HeadPose / EyeEAR
    │
    ▼
Fatigue
    │
    ▼
Engagement
    │
    ▼
Feature Engineering
    │
    ▼
XGBoost
    │
    ▼
SHAP
    │
    ▼
Dashboard
```

---

# 技术栈

## 计算机视觉

* OpenCV
* MediaPipe

---

## 数据分析

* NumPy
* Pandas

---

## 机器学习

* XGBoost
* Scikit-Learn

---

## 深度学习（后续）

* PyTorch（LSTM）

---

## Web

* Flask
* Jinja2
* ECharts

---

## 数据库

当前：

* SQLite

后续：

* MySQL
* InfluxDB

---

# 项目目录

```text
Student-EarlyWarning-System
│
├── docs/                项目文档
├── src/                 项目源码
├── datasets/            数据集
├── models/              模型文件
├── configs/             配置文件
├── tests/               测试代码
├── outputs/             输出结果
├── notebooks/           实验记录
├── assets/              项目资源
├── releases/            发布版本
│
├── requirements.txt
├── pyproject.toml
├── environment.yml
└── README.md
```

---

# 核心模块

| 模块                  | 功能      |
| ------------------- | ------- |
| Camera              | 摄像头采集   |
| Face Detection      | 人脸检测    |
| MediaPipe           | 关键点提取   |
| EyeEAR              | 闭眼检测    |
| HeadPose            | 头部姿态估计  |
| Fatigue             | 疲劳分析    |
| Engagement          | 课堂参与度分析 |
| Feature Engineering | 特征工程    |
| XGBoost             | 风险预测    |
| Dashboard           | 数据可视化   |

---

# 当前开发状态

| 模块        | 状态     |
| --------- | ------ |
| 项目规划      | ✅ 已完成  |
| 文档设计      | ✅ 已完成  |
| 数据库设计     | ✅ 已完成  |
| CV 模块设计   | ✅ 已完成  |
| AI 模块设计   | ✅ 已完成  |
| 项目框架      | 🚧 开发中 |
| Camera    | ⏳ 待开发  |
| MediaPipe | ⏳ 待开发  |
| Dashboard | ⏳ 待开发  |
| XGBoost   | ⏳ 待开发  |

---

# 开发流程

项目遵循 **"框架优先、接口优先、模块解耦、逐步完善"** 的开发策略。

开发流程如下：

```text
第一阶段
项目框架（Scaffold）

↓

第二阶段
CV模块开发

↓

第三阶段
课堂状态分析

↓

第四阶段
机器学习

↓

第五阶段
部署与优化
```

所有开发均围绕统一接口进行，各模块之间通过公开接口通信，降低耦合度，提高协作效率。

---

# 环境要求

推荐环境：

| 软件      | 版本    |
| ------- | ----- |
| Python  | 3.11+ |
| PyCharm | 2025+ |
| Git     | 最新稳定版 |

主要依赖：

```text
opencv-python
mediapipe
numpy
pandas
xgboost
scikit-learn
flask
matplotlib
pyyaml
pytest
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

# 快速开始

克隆项目：

```bash
git clone <repository-url>
```

进入项目目录：

```bash
cd Student-EarlyWarning-System
```

安装依赖：

```bash
pip install -r requirements.txt
```

启动项目：

```bash
python src/main.py
```

---

# 项目规范

项目采用统一开发规范。

包括：

* Git 规范
* Commit 规范
* Python 编码规范
* API 规范
* 日志规范
* 命名规范

详细内容请参考：

```text
docs/06-开发规范/
```

---

# 文档

所有设计文档均位于：

```text
docs/
```

主要包括：

* 项目概览
* 需求分析
* 总体设计
* 数据库设计
* CV 模块
* 机器学习
* 开发规范
* 测试
* 项目管理

---

# 开发计划

当前优先级：

```text
★★★★★

完成项目基础框架（Scaffold）
```

近期目标：

* 完成 `src/` 项目框架
* 实现 Camera 模块
* 建立 MediaPipe 数据流
* 完成 SQLite 数据存储
* 建立 Flask Dashboard

在完成基础框架后，再逐步实现各算法模块。

---

# 团队成员

本项目采用模块化协作方式开发。

| 成员 | 负责内容           |
| -- | -------------- |
| A  | 项目管理、架构设计      |
| B  | 计算机视觉          |
| C  | MediaPipe、人脸检测 |
| D  | 疲劳检测           |
| E  | 课堂参与度分析        |
| F  | 数据库            |
| G  | 后端与可视化         |
| H  | 测试与质量保障        |

具体分工请参考：

```text
docs/09-项目管理/成员分工.md
```

---

# License

本项目仅用于 SRTP 科研项目研究与教学实践。

版权所有 © 2026 Student-EarlyWarning-System Team.
