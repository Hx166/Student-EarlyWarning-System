# CHANGELOG

项目版本发布记录。

---

## v0.4.0

发布日期：2026-08-13

说明：第四阶段（机器学习）完成。

### 新增 / 完成

- 第四阶段特征工程完成：Feature Engineering 将状态/行为记录构建为统一 11 列特征向量，支持滑动窗口统计、归一化与 CSV 导出。
- XGBoost 模型完成：4 类状态分类（正常/注意力下降/疲劳/高风险）、风险概率、风险等级映射、训练/预测/评估/保存加载。
- 模型管理完成：版本化保存/加载、模型元数据、特征名记录、模型状态检查、统一模型获取入口。
- Risk Prediction / 预测推理完成：统一预测接口，输出风险类别、风险概率/置信度、风险等级。
- 第四阶段测试完成：特征工程、XGBoost、模型管理、风险预测测试均通过。
- 明确 LSTM 本阶段暂不开发，属后续版本。
- 明确 RiskPredictor 已作为独立预测/推理模块完成，暂未接入实时 Pipeline / Flask API；实时预测、在线预警、部署优化留待下一阶段。

### 测试

- tests/model/：特征工程、XGBoost、模型管理、Risk Prediction 测试全部通过。

---

## v0.3.0

发布日期：2026-08-05

说明：第一阶段到第三阶段完成，课堂状态分析与预警系统可运行、可演示。

### 新增 / 完成

- 第二阶段 CV 完成：Camera、FaceDetection、MediaPipe、EyeEAR、HeadPose 可实时产生视觉数据。
- 第三阶段 Fatigue：基于 PERCLOS、眨眼率、低头时长的加权融合评分与等级划分。
- 第三阶段 Engagement：基于眼部、头部姿态、疲劳与在场情况的加权参与度评分。
- Pipeline 联调：Camera → CV（Fatigue+Engagement） → SQLite 持久化（FrameRecord/StateRecord/WarningLog） → API → Dashboard 全链路打通。
- API 修复：注入数据库适配查询接口，修复 api-only 与主流程模式下数据库未连接问题；新增 /api/warnings、/api/behaviors、/api/frames 端点。
- MediaPipe 单实例：通过依赖注入共享给 FaceDetector/EyeEAR/HeadPose，避免重复初始化模型引擎。

### 测试

- 全量测试：54 passed, 3 skipped（skip 为需 cv2/mediapipe 环境的杰性跳过）。

---

## v0.1.0

说明：项目框架（Scaffold）与第一阶段完成，系统可启动并打通完整数据流。
