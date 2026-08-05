# CHANGELOG

项目版本发布记录。

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
