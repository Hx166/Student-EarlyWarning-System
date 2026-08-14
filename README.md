# 多模态视觉驱动的学生学业状态分析与预警系统

> SRTP 科研项目（Student Research Training Program）

基于 **OpenCV + MediaPipe + XGBoost + Flask** 的学生课堂行为分析与学业状态预警系统。

---

# 项目简介

本项目旨在构建一套基于计算机视觉（Computer Vision）的学生课堂状态分析平台，通过摄像头实时采集学生课堂行为信息，结合机器学习算法，对学生课堂参与度、疲劳状态及学习风险进行分析，为教师提供客观、实时、可解释的辅助决策依据。

系统采用模块化设计，将视觉特征提取、课堂行为分析、数据存储、可视化展示以及机器学习预测进行解耦，实现良好的可扩展性与可维护性。

---

# 项目目标

项目采用"框架优先、接口优先、模块解耦、逐步完善"的迭代开发策略。

截至 2026-08-14，`docs/2026-08-01起开发计划.md` 规划的四个开发阶段已全部完成：

| 阶段 | 主要内容 | 状态 |
|------|---------|------|
| 第一阶段 | 项目框架（Scaffold）：目录、配置、日志、Camera/CV/Database/API/Dashboard 框架、main 启动流程 | ✅ 已完成 |
| 第二阶段 | 计算机视觉：Camera、FaceDetection、MediaPipe、EyeEAR、HeadPose | ✅ 已完成 |
| 第三阶段 | 课堂状态分析：Fatigue、Engagement、SQLite 持久化、Flask API、Dashboard 实时展示 | ✅ 已完成 |
| 第四阶段 | 机器学习：Feature Engineering、XGBoost、Model Management、Risk Prediction | ✅ 已完成 |

下一阶段为「第五阶段：部署与优化」，尚未开始（详见文末「下一阶段计划」）。

已实现的核心能力：

* 摄像头实时采集
* 人脸检测（MediaPipe Face Mesh 468 关键点）
* 眼部 EAR 闭眼检测 / 眨眼 / PERCLOS
* Head Pose 姿态估计（Pitch/Yaw/Roll）
* 疲劳状态分析（Fatigue）
* 课堂参与度分析（Engagement）
* SQLite 数据存储
* Flask Dashboard 可视化
* 特征工程（Feature Engineering）
* XGBoost 4 类状态分类与学业风险预测
* 模型管理（版本化保存/加载）
* Risk Prediction 预测/推理模块

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
Dashboard
```

> 注：SHAP、LSTM 属后续版本（V1.0 / 第五阶段）内容，尚未开发。

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
├── configs/             配置文件（YAML）
├── tests/               测试代码（pytest）
├── scripts/             演示脚本
├── data/                SQLite 数据库文件
├── logs/                运行日志
├── models/              模型文件（训练/加载时生成，不入库）
├── datasets/            数据集（训练时生成，不入库）
│
├── requirements.txt
├── pyproject.toml
├── environment.yml
├── AGENTS.md
└── README.md
```

---

# 核心模块

| 模块 | 功能 | 状态 |
| ---- | ---- | ---- |
| Camera | 摄像头采集（OpenCV） | ✅ 已完成 |
| Face Detection | 人脸检测 | ✅ 已完成 |
| MediaPipe | 468 关键点 + Pose | ✅ 已完成 |
| EyeEAR | 闭眼 / 眨眼 / PERCLOS | ✅ 已完成 |
| HeadPose | 头部姿态估计（Pitch/Yaw/Roll） | ✅ 已完成 |
| Fatigue | 疲劳分析 | ✅ 已完成 |
| Engagement | 课堂参与度分析 | ✅ 已完成 |
| Feature Engineering | 特征工程 | ✅ 已完成（第四阶段） |
| XGBoost | 4 类状态分类 / 风险预测 | ✅ 已完成（第四阶段） |
| Model Management | 模型管理（版本化） | ✅ 已完成（第四阶段） |
| Risk Prediction | 风险预测 / 推理 | ✅ 已完成（第四阶段） |
| Dashboard | ECharts 实时看板 | ✅ 已完成 |

---

# 当前开发状态

| 模块 | 状态 |
| ---- | ---- |
| 第一阶段（项目框架） | ✅ 已完成 |
| 第二阶段（计算机视觉） | ✅ 已完成 |
| 第三阶段（课堂状态分析） | ✅ 已完成 |
| 第四阶段（机器学习） | ✅ 已完成 |
| 第五阶段（部署与优化） | ⏳ 待开发 |

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

# 四阶段完成情况

第一阶段–第四阶段已全部完成：

* 第一阶段：项目框架（Scaffold）完成，Camera/CV/Database/API/Dashboard/Config/Logger 框架可运行，`python src/main.py` 可启动完整数据流。
* 第二阶段：CV 模块完成（Camera、FaceDetection、MediaPipe、EyeEAR、HeadPose），视觉数据实时产生。
* 第三阶段：课堂状态分析完成（Fatigue、Engagement、SQLite 持久化、Flask API、Dashboard 实时展示）。
* 第四阶段：机器学习完成（Feature Engineering、XGBoost、Model Management、Risk Prediction），并完成相关测试。

---

# 已实现模块

| 模块 | 状态 |
|------|------|
| Camera | ✅ |
| FaceDetection | ✅ |
| MediaPipe（468 关键点 + Pose） | ✅ |
| EyeEAR（EAR/眨眼/PERCLOS） | ✅ |
| HeadPose（Pitch/Yaw/Roll） | ✅ |
| Fatigue（疲劳评分/等级） | ✅ |
| Engagement（参与度评分/等级） | ✅ |
| SQLite（FrameRecord/StateRecord/WarningLog） | ✅ |
| Flask API（/state /warnings /behaviors /frames） | ✅ |
| Dashboard（ECharts 实时看板） | ✅ |
| Feature Engineering（特征向量构建/归一化/CSV） | ✅ |
| XGBoost（4 类状态分类/风险预测/评估） | ✅ |
| Model Management（版本化保存/加载/元数据/注册） | ✅ |
| Risk Prediction（风险类别/概率/等级输出） | ✅ |

---

# 已知限制

* LSTM 时序预测、SHAP 可解释性属后续版本，本次（第四阶段）未开发。
* RiskPredictor 已作为独立预测/推理模块完成，尚未接入实时 Pipeline / Flask API。
* 实时预测、在线预警、部署优化留待下一阶段（第五阶段）。
* Fatigue 的哈欠（Yawn/MAR）特征尚未接入融合（Mouth MAR 模块仍为占位，恒返回 0）。
* 行为事件（blink/head_down/yawn/looking_away/absent）暂未在运行时写入 BehaviorRecord（表已建立，运行 Pipeline 仅写 FrameRecord/StateRecord/WarningLog）。
* MediaPipe Pose 关键点已计算但尚未被运行时链路消费。
* 多人场景下视觉分析仅处理第一张人脸（`CVProcessor` 取 `faces[0]`），逐人分析待后续。
* 多人/多摄像头同步、MySQL/InfluxDB、用户权限系统属后期计划。
* Dashboard 依赖 ECharts CDN，离线/内网环境需配置本地资源。
* 实时 FPS 为前端基于状态记录时间间隔估算，非后端精确框速。

---

# 下一阶段计划（第五阶段：部署与优化）

* 实时预测接入在线预警链路。
* Flask API 提供预测接口。
* SHAP 可解释性、LSTM 时间序列预测。
* 补齐 MAR/打哈欠与行为事件持久化。
* MySQL/InfluxDB、Docker 部署与性能优化。

---

# 环境要求

推荐环境：

| 软件 | 版本 |
| ---- | ---- |
| Python | 3.11+ |
| PyCharm | 2025+ |
| Git | 最新稳定版 |

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

# 团队成员

本项目采用模块化协作方式开发。

| 成员 | 负责内容 |
| ---- | -------- |
| A | 项目管理、架构设计 |
| B | 计算机视觉 |
| C | MediaPipe、人脸检测 |
| D | 疲劳检测 |
| E | 课堂参与度分析 |
| F | 数据库 |
| G | 后端与可视化 |
| H | 测试与质量保障 |

具体分工请参考：

```text
docs/09-项目管理/成员分工.md
```

---

# License

本项目仅用于 SRTP 科研项目研究与教学实践。

版权所有 © 2026 Student-EarlyWarning-System Team.
