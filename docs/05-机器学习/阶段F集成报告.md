# 阶段 F 集成报告

> 项目：Student Early Warning System
>
> 阶段：阶段 F（engagement-v1 运行时集成）
>
> 完成日期：2026-10-31
>
> 模型：`models/xgboost/engagement-v1/`

---

## 1. 配置改动

`configs/system.yaml` 的 `prediction` 段：

| 配置 | 修改前 | 修改后 |
|---|---|---|
| model_version | `"v1"` | `"engagement-v1"` |
| window_seconds | `30` | `10` |
| min_records | `30` | `10` |

`model_name: "xgboost"` 与 `enabled: true` 保持不变，其余配置键未修改。

## 2. 运行时接线

### PredictionService

- 新增可选末尾参数 `model_version`，不改变旧调用形式。
- `_ensure_predictor` 将版本传入 `RiskPredictor`。
- 新增只读属性 `last_probabilities`，返回最近一次 `level_0..level_3` 四类概率快照。
- `predict` 调用新增的 `predict_with_proba`，成功后保存概率。
- 为满足 11 列训练/在线一致性，`predict` 新增可选 `behavior_records`；旧单参数调用保持兼容。

### RiskPredictor

- 新增 `predict_with_proba(features) -> (RiskPredictionResult, np.ndarray)`。
- 单行预测和批量预测复用同一内部计算逻辑。
- 原有 `predict`、`predict_batch` 的输入和返回结构不变。

### PipelineService

- 读取 `prediction.model_version` 并传入 `PredictionService`。
- 新增只读属性 `last_prediction`，字段为：
  - `record_time`
  - `model_name`
  - `model_version`
  - `level`
  - `probability`
  - `probabilities`
- 缓存已经完成的行为事件，使在线特征构建同时获得 StateRecord 和 BehaviorRecord。
- Engagement 结果写入 `PredictionLog` 时，`risk_level` 使用相同的 `level_*` 名称。
- Engagement 不触发 `warning/high_risk` 模型预警；原有疲劳、低参与度和无人脸规则预警保持不变。

### API

- `get_risk_predictor` 向 `RiskPredictor` 传入 `model_version=engagement-v1`。
- 新增 `GET /api/prediction/latest`。
- `/api/predict` 对 Engagement 返回值将 `risk_level` 规范为 `level_*`；响应字段未删除。
- `/api/model/info` 增加顶层 `label_mapping` 字段，同时保留原有元数据。

## 3. 接口验证

实际启动：

```powershell
python -m src.main --api-only
```

服务成功监听本机 5000 端口，以下请求均由真实 HTTP 调用完成。

### GET /api/model/info

摘要：

```json
{
  "code": 0,
  "data": {
    "name": "xgboost",
    "version": "engagement-v1",
    "feature_names": [
      "ear_mean", "ear_std", "blink_rate", "perclos",
      "pitch_mean", "yaw_mean", "roll_mean", "fatigue_score",
      "attention_score", "head_down_duration", "face_lost_duration"
    ],
    "label_mapping": {
      "0": "level_0",
      "1": "level_1",
      "2": "level_2",
      "3": "level_3"
    }
  },
  "message": "ok"
}
```

### POST /api/predict

使用 `daisee_features.csv` 第一条样本的 11 个特征：

```json
{
  "code": 0,
  "data": {
    "risk_label": "level_2",
    "risk_level": "level_2",
    "risk_probability": 0.5444582104682922,
    "confidence": 0.5444582104682922,
    "model_version": "engagement-v1",
    "extra": {}
  },
  "message": "ok"
}
```

旧字段 `risk_label`、`risk_probability`、`risk_level` 和 `model_version` 均保留。

### GET /api/prediction/latest

API-only 模式没有摄像头流水线，实测返回：

```json
{
  "code": 0,
  "data": null,
  "message": "ok"
}
```

使用假 Pipeline 的接口测试验证了有预测时的数据结构和 null 降级，两种请求均为 HTTP 200。

## 4. 三方特征一致性

运行脚本：

```powershell
python scripts/verify_feature_parity.py
```

对 `stage_c_sample.csv` 前 3 条片段重新完整解码，并比较：

1. `FeatureBuilder(10)` + state/behavior 训练侧；
2. `PredictionService` 的 state/behavior 在线侧；
3. `daisee_features.csv` 冻结特征表。

| clip_id | 最大绝对差异 | 结果 |
|---|---:|---|
| 1100152070.avi | 8.881784197e-16 | PASS |
| 1110031063.avi | 1.42108547152e-14 | PASS |
| 2100582056.avi | 8.881784197e-16 | PASS |

整体最大差异 `1.42108547152e-14`，低于阈值 `1e-9`。明细共 33 行，保存在 `outputs/stage_f/feature_parity.csv`。

首次验证发现在线服务原先只接收 StateRecord，导致 `blink_rate` 与 `face_lost_duration` 为 0、最大差异达到 18–24。补齐行为事件传递后重新完整解码验证，三方一致。

## 5. 前端展示

`index.html` 新增“模型参与度”卡片，不改变其他卡片结构。

`app.js` 每 2 秒刷新时与 state、warning、FPS 并行请求 `/api/prediction/latest`，展示：

- 等级：`level_0` 很低、`level_1` 较低、`level_2` 较高、`level_3` 很高；
- 当前等级置信度；
- 模型版本；
- 四类概率横条。

无数据、请求失败或模型未就绪时显示“模型未就绪”，四类概率为 0%，不覆盖原有指标、图表、预警和表格逻辑。

`style.css` 只新增模型卡片及概率条样式。`node --check app.js` 通过。

## 6. 测试与运行结果

新增测试：

- `tests/service/test_prediction_version.py`
- `tests/api/test_prediction_latest.py`
- `tests/api/test_model_info.py`

覆盖模型版本传递、buffer 特征一致、行为特征传递、Engagement 无模型风险预警、最新预测接口和模型信息。

API/Service 回归集：

```text
40 passed
```

3 条视频完整处理与三方校验耗时约 22.6 秒，临时目录已自动清理。

## 7. 已知限制

- engagement-v1 是基线模型，Test accuracy 0.494395 仅与多数类基线持平，尚未证明优于基线。
- Level 0 在 Test 中只有 4 条，模型对其识别能力不足。
- `risk_level` 字段因兼容性保留，但 Engagement 模式下其值等于 `level_*`，不表示风险等级。
- 四类概率条用于辅助理解模型输出，不应当作校准后的现实信心保证。
- 实时行为事件要等待事件结束后才能进入窗口；进行中的行为事件不会被提前计入，这与离线记录语义一致。

## 8. 结论

阶段 F 的运行时接线、独立标签映射、最新预测接口、前端展示和模型信息接口均已完成。真实 API 服务启动验证成功，三条片段的训练侧、在线侧和冻结特征表最大差异满足 `1e-9` 门槛。

本阶段未重训模型、未修改数据、未改变风险 `LABEL_MAPPING` 或 `RiskPredictor.predict` 行为、未修改 API 请求结构、未修改数据库结构，也未进入阶段 G。

---

## 9. 验收记录（规划侧，2026-10-31）

- **验收结论**：通过。配置、接口、一致性、前端与测试均可复现。
- **接口复核**（Flask test_client）：`/api/model/info` 返回 `engagement-v1` 与 `level_0..3`；`/api/predict` 对特征表首行返回 `level_2`、概率 0.544458；`/api/prediction/latest` 在无流水线时返回 null。
- **一致性复核**：`outputs/stage_f/feature_parity.csv` 33 行、3 条片段，最大差异 1.42e-14，与报告一致。
- **测试**：API/Service 集 38 passed、2 errors；2 项错误均为 `tmp_path` 临时目录权限，不是断言失败。
- **前端**：`index.html` 模型卡片存在，`node --check app.js` 通过。
- **遗留**：`_recent_behaviors` 只追加不裁剪，长会话内存缓慢增长；阶段 G 顺手裁剪到最近 60 秒。`risk_level` 仅为兼容字段，展示不使用。
