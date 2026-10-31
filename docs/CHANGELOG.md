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
| v1.1.0 | 第六阶段：DAiSEE Engagement 四分类（进行中） |

---

## v1.1.0-dev（进行中）

更新日期：2026-10-28

说明：第六阶段（数据集与训练）启动。计划与验收标准见 `docs/2026-10-28起开发计划.md`，数据规范见 `docs/05-机器学习/数据集与标注.md`，V1 决策见 `docs/09-项目管理/决策记录.md`。

### 已完成（2026-09-17 ~ 2026-10-28）

- 离线处理链路：`VideoClock`（视频时间轴替代墙钟）、`OfflineFatigueAnalyzer`（离线疲劳适配）、`OfflineVideoProcessor`（视频 → 秒级 StateRecord / BehaviorRecord → 特征 CSV）。
- PoC 脚本：`scripts/sample_videos.py`、`scripts/offline_poc_run.py`、`scripts/offline_poc_report.py`、`scripts/poc_labels.py`。
- 特征修复：新增 `STAT_TO_FEATURE_MAPPING`，`avg_ear_mean` 等 8 个聚合列正确映射为 `ear_mean` 等特征向量列。
- 测试：新增 11 项离线 pytest，全项目测试用例 181 项。

### 2026-10-28 数据核查（只读）

- DAiSEE 位于 `D:\数据集\DAiSEE\DAiSEE`，仓库内 `datasets/raw/daisee` 为空，数据未复制。
- 9067 个视频（.avi 8231 / .mp4 836）、8571 条标签记录（可关联 8570）、113 名被试；官方三集被试无交集。
- Engagement 分布：0: 61、1: 440、2: 4312、3: 3758。
- 497 个视频无标签、1 条标签无视频；全量元数据均为 640×480 / 30fps，帧数 300 帧 8633 条，302 帧 214 条，314 帧 219 条，288 帧 1 条。

### 阶段 B 完成（2026-10-28）

- 新增 `scripts/check_daisee.py`：全量元数据探测、每 split 抽样首帧解码、标签关联、manifest 与异常清单导出。
- 产出 `datasets/processed/daisee_manifest.csv`（9067 行 × 16 列）、`daisee_issues.csv`（932 行：无标签 497、孤立标签 1、帧数偏差 434）。
- 报告 `docs/05-机器学习/DAiSEE数据核查报告.md`；测试 `tests/offline/test_daisee_manifest.py`（1 项）。
- 抽样解码 60/60 通过；未发现坏文件；原始数据集未被写入。

### 阶段 C 完成（2026-10-28）

- 新增 `scripts/stage_c_features.py`：固定抽样、单条试跑、批量处理、A/B 聚合对比与特征导出。
- 产出 `datasets/processed/stage_c_sample.csv`（20 条）、`stage_c_agg_compare.csv`（220 行）、`datasets/features/stage_c_features.csv`（20 行）。
- 报告 `docs/05-机器学习/阶段C特征验证报告.md`；测试 `tests/offline/test_stage_c_features.py`（2 项）。
- 结果：20/20 片段成功，NaN 0、无常量特征；18/20 条 A/B 一致，2 条异常帧片段存在窗口边界差异；单条均值 11.656 秒，全量约 27.75 小时。
- 决策：聚合统一取 A（D-008）；`--mode all` 与 `training.target` 两处小修移至阶段 D（D-009）。

### 阶段 D 完成（2026-10-31）

- 新增 `scripts/build_daisee_features.py`：多进程提取、原子断点续跑、失败率门禁、合并校验与统计。
- 产出 `datasets/features/daisee_features.csv`（8570 行 × 20 列，SHA256 前 16 位 `63c0b6ffacb7dd93`）、`daisee_features_stats.csv`、`outputs/daisee_extract/{timings,failures}.csv`。
- 报告 `docs/05-机器学习/阶段D特征表报告.md`；测试 `tests/offline/test_build_daisee_features.py`（2 项）。
- 结果：split 5357/1429/1784，四等级 61/440/4312/3757，11 特征 NaN 0、无常量列、失败 0；与 C 的 20 条最大差异 1.42e-14；6/4 workers 活跃耗时约 4.59 小时。
- 异常：31 条 Test 片段实际 EOF 为 293/294/296 帧（文件头 300），按实际可解码帧记录；1 条 288 帧 Validation 片段无标签，未进入特征表。
- 前置小修完成：`stage_c_features.py` 统一 A 口径与 `--mode all`，`training.target=engagement`。

### 阶段 E 完成（2026-10-31）

- 新增 `src/models/labels.py`：Engagement 独立映射 `level_0`–`level_3`，不复用风险 `LABEL_MAPPING`。
- 新增 `scripts/train_engagement.py`：训练、评估、多数类基线、类别加权、消融、模型注册与加载验证。
- 模型 `models/xgboost/engagement-v1/`（6 个文件，含 metrics.json）；产物在 `outputs/stage_e/`。
- 报告 `docs/05-机器学习/阶段E训练评估.md`；测试 `tests/model/test_engagement_{labels,training}.py`（6 项）。
- 结果：Test accuracy 0.4944（=多数类基线），macro-F1 0.2581、weighted-F1 0.4827、AUC 0.6351；Level 0（n=4）全部误判；9 特征消融在 Test 全指标下降。
- 加权模型 macro-F1 0.2797，但 accuracy 降至 0.4613，未作为默认模型保存。
- 性能门槛“Test accuracy 高于多数类基线”未通过，按 R9 交付基线，不调低标准。

### 阶段 F 完成（2026-10-31）

- 运行时接入 `engagement-v1`：`configs/system.yaml` 版本、10 秒窗口、10 条记录。
- `PredictionService` 支持模型版本与行为事件；`RiskPredictor` 新增 `predict_with_proba`；`PipelineService` 暴露 `last_prediction`，Engagement 写入 `level_*` 且不触发风险预警。
- 新增 `GET /api/prediction/latest`；`/api/model/info` 增加 `label_mapping`；`/api/predict` 返回 level。
- 前端新增“模型参与度”卡片：等级、置信度、四类概率条、模型版本。
- 3 条片段训练侧/在线侧/特征表最大差异 1.42e-14（`scripts/verify_feature_parity.py`）。
- 报告 `docs/05-机器学习/阶段F集成报告.md`；API/Service 测试 40 项（本会话 38 通过、2 项临时目录权限未执行）。

### 阶段 G 完成（2026-10-31）

- 三页演示：实时看板 `/`、视频回放 `/playback`、模型说明 `/model`，统一导航。
- 新增 `src/api/routes/demo.py`、`src/service/demo_playback.py`：20 条固定片段、服务端处理、LRU 缓存、MP4 受控访问。
- 回放实测：`1100152070.avi` 预测 level_0（0.5778），二次请求 0.007 秒；`401835015.mp4` 预测 level_2（0.4462）。
- 模型页读取阶段 E 指标与元数据；缺失时显示“指标未生成”。
- `PipelineService._recent_behaviors` 裁剪到最近 60 秒。
- 报告 `docs/05-机器学习/阶段G演示报告.md`、`docs/10-成果/演示手册.md`。
- 测试 47 项（本会话 44 通过、3 项临时目录权限未执行）；`node --check` 三个脚本通过。
- 实时画面仍为占位，登记为 P1-01（`docs/01-需求分析/实时画面需求.md`），不阻塞 V1。

### 已确认阻塞

- Level 0 仅 61 条（Test 4 条），该类指标不可靠。
- 模型 Test accuracy 与多数类基线持平，演示按“基线模型”标注。

### 下一阶段计划

- P1-01 实时画面（MJPEG + 状态降级），预计 1–1.5 天；
- 其后按用户确认决定是否做疲劳/时序扩展。

---

## v1.0.0

发布日期：2026-08-19

说明：第五阶段（部署与优化）第 4 步完成，系统演进为可部署、可上线版本。

### 新增 / 完成

- Docker 部署：Dockerfile（Python 3.11 + 依赖 + 源码）、docker-compose.yml（App + MySQL + InfluxDB 可选）。
- 前端离线化：ECharts 本地资源（`src/frontend/static/js/echarts.min.js`），移除 CDN 依赖。
- 多人逐人分析：CVProcessor 对所有检测人脸逐人分析（FaceAnalysis），主脸结果保留顶层字段向后兼容。
- 后端精确 FPS：PipelineService 统计处理帧率，新增 GET /api/metrics/fps，前端改为消费后端 FPS。
- 文档收尾：填充 docs/07-部署/（安装说明/环境配置/Docker/发布流程）与 docs/08-测试/（单元/集成/性能/Bug记录）；更新 README、任务看板。

### 测试

- 新增多人逐人分析、后端 FPS 相关测试。
- 全量测试：160 passed, 8 skipped（8 个 skip 为可选依赖未安装：cv2、xgboost、torch）。

### 已知限制（收尾固化）

- Pipeline 采集/视觉/预测/写入尚未多线程/异步解耦（单线程串行）。
- InfluxDB 仅预留接口（默认关闭），未接入真实时序写入。
- Dashboard 的 SHAP 风险解释面板尚未接入（ShapExplainer 模块已完成）。
- GPU 推理未接入（默认 CPU）。
- 多人逐人分析已在 CV 层完成，数据库/预警链路仍为单学生。

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
- 全量测试：154 passed, 8 skipped（6 个 cv2/mediapipe 环境用例，2 个 torch 未安装跳过的 LSTM 集成用例）。

### 待完成（第 4 步）

- 见 v1.0.0（Docker 部署、ECharts 本地化、多人逐人分析、后端精确 FPS、文档收尾）。

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
