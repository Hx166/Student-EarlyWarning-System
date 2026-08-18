# CHANGELOG

项目版本发布记录。

版本号与四阶段开发计划对应关系：

| 版本 | 阶段 |
|------|------|
| v0.1.0 | 第一阶段（项目框架） |
| v0.2.0 | 第二阶段（计算机视觉） |
| v0.3.0 | 第三阶段（课堂状态分析） |
| v0.4.0 | 第四阶段（机器学习） |
| v0.5.0 | 第五阶段中间成果（实时预测接线 + 预测 API + SHAP） |
| v1.0.0 | 第五阶段完成（LSTM + 数据库迁移 + Docker + 性能优化） |

---

## v0.5.0

发布日期：2026-08-15

说明：第五阶段（部署与优化）第 1~3 步完成，系统具备在线预测、可解释性与时序预测能力。

### 新增 / 完成

- 实时预测接线：PredictionService 接入 Pipeline，FeatureBuilder → RiskPredictor → PredictionLog，warning/high_risk 触发预警；无模型时自动降级跳过。
- 预测 API：POST /api/predict、GET /api/predictions、GET /api/model/info。
- PredictionLog 表 + 索引（CREATE TABLE IF NOT EXISTS，重启自动建表）。
- MAR / 打哈欠：mouth_mar 实现真实 MAR（6 关键点），Fatigue 接入 Yawn 分量（权重 0.50/0.20/0.15/0.15）。
- 行为事件：BehaviorEventTracker 状态机，blink/head_down/yawn/looking_away/absent 运行时写入 BehaviorRecord；Pose 关键点接入离开/在场判断。
- SHAP：ShapExplainer 全局/局部解释，新增 shap 依赖。
- LSTM：LSTMModel（PyTorch，训练/推理/保存/加载），经 ModelRegistry 统一管理，新增 torch 依赖。
- 数据库后端抽象：DatabaseService 委托 DatabaseBackend（SQLite 默认 / MySQL 可切换 / InfluxDB 预留），新增 pymysql 依赖；提供 SQLite→MySQL 迁移脚本。

### 测试

- 第五阶段第 1~3 步新增测试：第 1 步 14 例、第 2 步 19 例、第 3 步 15 例。
- 全量测试：156 passed, 6 skipped（6 个 cv2/mediapipe 环境用例）。

### 待完成（第 4 步）

- Docker 部署、ECharts 本地化、多人逐人分析、多线程 Pipeline、后端精确 FPS、docs/08-测试 填充、文档收尾。

---

## v0.4.0

发布日期：2026-08-13

说明：第四阶段（机器学习）完成。

### 新增 / 完成

- 第四阶段特征工程完成：Feature Engineering 将状态/行为记录构建为统一 11 列特征向量，支持滑动窗口统计、归一化与 CSV 导出。
- XGBoost 模型完成：4 类状态分类（正常/注意力下降/疲劳/高风险）、风险概率、风险等级映射、训练/预测/评估/保存加载。
- 模型管理完成：版本化保存/加载、模型元数据、特征名记录、模型状态检查、统一模型获取入口。
- Risk Prediction / 预测推理完成：统一预测接口，输出风险类别、风险概率/置信度、风险等级。
- 明确 LSTM 本阶段暂不开发，属后续版本。
- 明确 RiskPredictor 已作为独立预测/推理模块完成，暂未接入实时 Pipeline / Flask API；实时预测、在线预警、部署优化留待下一阶段。

### 测试

- tests/model/：特征工程、XGBoost、模型管理、Risk Prediction 测试全部通过。
- 第四阶段完成后，全项目测试函数总数约 112 个（tests/ 下 17 个测试文件，含需 cv2/mediapipe 环境的跳过用例）。

### 已知限制（收尾固化）

- LSTM、SHAP 未开发，属后续版本。
- RiskPredictor 未接入实时 Pipeline / Flask API。
- Fatigue 哈欠（Yawn/MAR）特征未接入融合（Mouth MAR 模块仍为占位）。
- 行为事件（blink/head_down/yawn/looking_away/absent）暂未在运行时写入 BehaviorRecord。
- MediaPipe Pose 关键点已计算但尚未被运行时链路消费。

---

## v0.3.0

发布日期：2026-08-05

说明：第三阶段（课堂状态分析）完成（含第一、二阶段成果，系统可运行、可演示）。

### 新增 / 完成

- 第三阶段 Fatigue：基于 PERCLOS、眨眼率、低头时长的加权融合评分与等级划分。
- 第三阶段 Engagement：基于眼部、头部姿态、疲劳与在场情况的加权参与度评分。
- Pipeline 联调：Camera → CV（Fatigue+Engagement） → SQLite 持久化（FrameRecord/StateRecord/WarningLog） → API → Dashboard 全链路打通。
- API 修复：注入数据库适配查询接口，修复 api-only 与主流程模式下数据库未连接问题；新增 /api/warnings、/api/behaviors、/api/frames 端点。
- MediaPipe 单实例：通过依赖注入共享给 FaceDetector/EyeEAR/HeadPose，避免重复初始化模型引擎。

### 测试

- 全量测试：54 passed, 3 skipped（skip 为需 cv2/mediapipe 环境的跳过）。

---

## v0.2.0

发布日期：2026-08（与 v0.3.0 合并发布，回溯补记）

说明：第二阶段（计算机视觉）完成。

### 新增 / 完成

- Camera、FaceDetection、MediaPipe（468 关键点 + Pose）、EyeEAR（EAR/眨眼/PERCLOS）、HeadPose（Pitch/Yaw/Roll）可实时产生视觉数据。

---

## v0.1.0

说明：项目框架（Scaffold）与第一阶段完成，系统可启动并打通完整数据流。
