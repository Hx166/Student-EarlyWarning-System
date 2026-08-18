# API 规范

## 1. 概述

系统对外提供 REST 接口，供前端 Dashboard 与第三方调用。所有接口遵循统一的请求与响应格式。

- 默认监听：`http://<host>:5000`
- 数据格式：`application/json`
- 字符编码：UTF-8

## 2. 统一响应格式

所有接口返回统一 JSON 结构：

```json
{
    "code": 0,
    "data": {},
    "message": "ok"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 0 表示成功，非 0 表示失败 |
| data | any | 业务数据（成功时返回，失败时为 null 或 []） |
| message | string | 提示信息 |

## 3. 错误码约定

| HTTP 状态码 | code | 含义 |
|-------------|------|------|
| 200 | 0 | 成功 |
| 400 | 1 | 参数错误 / 特征校验失败 |
| 503 | 1 | 模型不可用 / 服务不可用 |

失败时 `message` 应给出可读原因，例如 `"缺少 features 字段"`、`"特征校验失败: 特征缺少必需列"`、`"模型不可用"`。

## 4. 现有接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/health | 健康检查 |
| GET | /api/state | 最新课堂状态数据（limit 参数） |
| GET | /api/metrics/summary | 课堂指标汇总 |
| GET | /api/warnings | 预警记录（limit、student_id） |
| GET | /api/behaviors | 行为事件记录（limit） |
| GET | /api/frames | 原始帧数据（limit） |

## 5. 预测接口

### 5.1 POST /api/predict

在线风险预测。输入单个特征向量，输出风险类别、概率与等级。

请求体：

```json
{
    "features": {
        "ear_mean": 0.29,
        "ear_std": 0.02,
        "blink_rate": 18.0,
        "perclos": 0.12,
        "pitch_mean": 8.5,
        "yaw_mean": 1.0,
        "roll_mean": 0.5,
        "fatigue_score": 20.0,
        "attention_score": 86.0,
        "head_down_duration": 0.0,
        "face_lost_duration": 0.0
    }
}
```

响应：

```json
{
    "code": 0,
    "data": {
        "risk_label": "normal",
        "risk_probability": 0.9,
        "risk_level": "normal",
        "confidence": 0.9,
        "model_version": "v1",
        "prediction_time": "2026-08-15T10:00:00",
        "extra": {}
    },
    "message": "ok"
}
```

说明：

- `features` 必须包含特征工程定义的 11 列特征（`FEATURE_VECTOR_COLUMNS`），列名与顺序固定；
- `features` 不得包含 `label` 列；
- 缺列或含 `label` 列时返回 HTTP 400；
- 模型文件不存在或未训练时返回 HTTP 503。

### 5.2 GET /api/predictions

查询历史预测记录。

参数：`limit`（默认 50）、`student_id`（可选）。

响应 `data` 为 PredictionLog 记录数组，字段包括 `prediction_id`、`student_id`、`session_id`、`record_time`、`risk_label`、`risk_probability`、`risk_level`、`model_version`、`feature_json`。

### 5.3 GET /api/model/info

查询当前加载模型信息。

响应 `data` 字段：

| 字段 | 说明 |
|------|------|
| name | 模型名称 |
| version | 模型版本 |
| exists | 模型文件是否存在 |
| loaded | 是否已加载到内存 |
| path | 模型版本目录 |
| metadata | 模型元数据（版本/特征名/类型/保存时间） |
| feature_names | 特征列名列表 |
