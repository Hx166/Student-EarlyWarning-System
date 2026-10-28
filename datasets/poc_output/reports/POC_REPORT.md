======================================================================
离线处理 PoC —— 最终报告
======================================================================

【修改文件】全部为新增，未修改任何现有生产代码：
- src/cv/behavior/clock.py          (新) 视频时间轴 VideoClock 与 mock 注入/卸载
- src/cv/behavior/offline_fatigue.py(新) 复用 FatigueAnalyzer 的离线适配器
- src/service/offline_video_processor.py(新) 视频→StateRecord/BehaviorRecord→特征 CSV
- scripts/sample_videos.py          (新) 合成/抽样 PoC 视频
- scripts/offline_poc_run.py        (新) 小样本运行器
- scripts/offline_poc_report.py     (新) 汇总报告生成
- scripts/poc_labels.py             (新) 标签映射生成
- tests/offline/                    (新) 11 项 pytest

【PoC 运行方式】
  conda activate Student-EarlyWarning-System
  python scripts/poc_labels.py
  python scripts/sample_videos.py
  python scripts/offline_poc_run.py --max_frames 300
  python scripts/offline_poc_report.py --output_dir datasets/poc_output

【测试结果】181 passed（新增 11 项 + 既有 170 项），1 个既有 FutureWarning。

【特征统计/视频参数】见 datasets/poc_output/reports/poc_summary.txt：
- 5 视频全部 30fps、640x480、每段 10s、300 帧；每段 10 条 StateRecord。
- Face Mesh 检出率 98.7%~100%；drowsy 视频 face_lost=1.3%。
- ear_mean 0.333~0.483（drowsy 最低）；ear_std 0.0145~0.055。
- pitch_mean 6.3~22.3（低头主导疲劳）；yaw -14.3~-8.8；roll 4.7~6.5。
- fatigue_score 0~6.5（半窗口 + 限帧导致偏低）；attention_score 80~98。
- blink_rate / face_lost_duration 在合成数据上为 0（见问题 1、2）。
- DAiSEE vs UTA：ear 0.473 vs 0.439、pitch 12.5 vs 18.5、fatigue 1.2 vs 1.7 → 存在明显 Domain Shift。

【发现的问题】
1. 合成视频眨眼/EAR 无法触发 ear_threshold=0.20（探针测得合成闭眼 EAR≈0.265），
   blink_rate/perclos 在合成数据上恒 0；需用真实人脸视频验证（待真实数据）。
2. 半窗口：DAiSEE 10s 视频在 30s 窗口下所有样本都是半窗口（record range 9s），
   前 1s 样本 ear_std 为 NaN（pandas 单样本 std），模型端可能产生 NaN 泄漏。
3. 相邻窗口高度重叠：window step=1s，30s 窗口 → 每视频 10 个样本几乎全同特征（伪重复）。
4. 离线疲劳已按视频时间工作（测试验证 10s 低头 → head_down≈10s），不再受墙钟影响。
5. mediapipe 需 0.10.14；当前 base 环境 Python 3.13 无该版本，需用项目环境
   Student-EarlyWarning-System (Python 3.10.20 + mediapipe 0.10.14 + xgboost 3.2.0)。
6. UTA 的 <30fps 与真实 blink 检测可能需要降低 blink_min_frames 或校准 EAR 阈值（待真实数据）。

【下一阶段建议】
A. 训练窗口方案（不修改生产）：
   - 建议离线训练采用「固定不重叠窗口」：segmentation 每 30s 一段（DAiSEE 需拼接/复制片段或放弃
     完整 30s 语义），避免 rolling 半窗口与伪重复；
   - 在线推理仍用现有 FeatureBuilder rolling(30s, min_periods=1) 与 PredictionService。
B. 将 source_dataset/subject_id/video_id 仅作元数据，不入 XGBoost 特征（已如此实现）。
C. 推荐标签子集：DAiSEE→normal/attention_down；UTA→normal/attention_down/fatigue，
   不生成 high_risk。
D. 真实数据 PoC：下载 3~5 个 DAiSEE + 3 个 UTA 片段（各 5~10 秒），放置 datasets/raw/daisee
   与 datasets/raw/uta_rldd 后重跑 sample_videos.py（自动从 raw 抽样，不再生成合成视频），
   并补充真实 ear_std/blink_rate/perclos 验证。
E. 待验证：真实视频的 FPS/分辨率/标签 CSV 列名；UTA 被试互斥与授权；DAiSEE 标签分布。
======================================================================
