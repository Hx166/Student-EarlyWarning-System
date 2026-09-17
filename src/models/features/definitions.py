"""src/models/features/definitions.py —— 特征工程公共定义。

集中定义特征列名、统计聚合规则与行为事件到特征列的映射，
保证 Feature Engineering 模块与特征文档命名一致，避免同一特征出现多个名称。
"""

from __future__ import annotations

from typing import Dict, List


# StateRecord 连续数值特征（原始特征定义）
STATE_FEATURE_COLUMNS: List[str] = [
    "avg_ear",
    "avg_mar",
    "avg_pitch",
    "avg_yaw",
    "avg_roll",
    "perclos",
    "attention_score",
    "fatigue_score",
]

# 每个连续特征生成的统计列（mean/std/min/max/median）
AGGREGATION_SUFFIXES: List[str] = ["mean", "std", "min", "max", "median"]

# 最终特征向量列顺序（与 docs/05-机器学习/XGBoost.md 第4节输入特征表一致）
FEATURE_VECTOR_COLUMNS: List[str] = [
    "ear_mean",
    "ear_std",
    "blink_rate",
    "perclos",
    "pitch_mean",
    "yaw_mean",
    "roll_mean",
    "fatigue_score",
    "attention_score",
    "head_down_duration",
    "face_lost_duration",
]

# 统计特征列（由 8 个连续特征 × 5 个聚合统计量生成，共 40 列）
STATISTICAL_FEATURE_COLUMNS: List[str] = [
    f"{col}_{suffix}"
    for col in STATE_FEATURE_COLUMNS
    for suffix in AGGREGATION_SUFFIXES
]

# 可供统计聚合且单位明确的列（排除 record_time 等时间元数据）
NUMERIC_STATE_COLUMNS: List[str] = [
    col for col in STATE_FEATURE_COLUMNS if col not in ("record_time",)
]

# 行为事件类型与生成特征的映射
BEHAVIOR_TYPE_COLUMNS: Dict[str, str] = {
    "blink": "blink_count",
    "head_down": "head_down_count",
    "looking_away": "face_lost_count",
    "absent": "face_lost_count",
    "yawn": "yawn_count",
}

# 行为事件类型到特征列的映射（用于特征向量构造）：
# head_down_duration 由 head_down 事件 duration 求和生成
# face_lost_duration 由 looking_away/absent 事件 duration 求和生成
BEHAVIOR_STAT_COLUMNS: Dict[str, List[str]] = {
    "blink": ["blink_rate"],
    "head_down": ["head_down_duration", "head_down_count"],
    "looking_away": ["face_lost_duration", "face_lost_count"],
    "absent": ["face_lost_duration", "face_lost_count"],
    "yawn": ["yawn_count"],
}

# 特征向量中的统计行为列（除基础统计列外）
BEHAVIOR_FEATURE_COLUMNS: List[str] = [
    "blink_rate",
    "head_down_duration",
    "face_lost_duration",
]

# 全部统计特征列（统计特征 + 行为特征）
FINAL_STATISTICAL_COLUMNS: List[str] = STATISTICAL_FEATURE_COLUMNS + BEHAVIOR_FEATURE_COLUMNS

# 特征向量中使用的统计列（特征向量同时包含统计特征列与基础特征列）
FEATURE_VECTOR_STAT_COLUMNS: List[str] = [
    "ear_mean",
    "ear_std",
    "pitch_mean",
    "yaw_mean",
    "roll_mean",
]

# 特征向量中的原始（每窗口）特征列
FEATURE_VECTOR_RAW_COLUMNS: List[str] = [
    "perclos",
    "fatigue_score",
    "attention_score",
]

# 最终特征向量列顺序（与 XGBoost 文档第 4 节特征表一致）

# 特征向量中各统计列的聚合函数
WINDOW_STAT_FUNCTIONS: Dict[str, str] = {
    "ear_mean": "mean",
    "ear_std": "std",
    "perclos": "mean",
    "pitch_mean": "mean",
    "yaw_mean": "mean",
    "roll_mean": "mean",
    "fatigue_score": "mean",
    "attention_score": "mean",
}

# 聚合统计列到最终特征向量列的映射
# 说明：特征向量列名（ear_mean/pitch_mean 等）去掉了原始状态列的 avg_ 前缀，
# 而 perclos/fatigue_score/attention_score 保留原名，无法用简单拼接推导，需显式映射。
STAT_TO_FEATURE_MAPPING: Dict[str, str] = {
    "avg_ear_mean": "ear_mean",
    "avg_ear_std": "ear_std",
    "perclos_mean": "perclos",
    "avg_pitch_mean": "pitch_mean",
    "avg_yaw_mean": "yaw_mean",
    "avg_roll_mean": "roll_mean",
    "fatigue_score_mean": "fatigue_score",
    "attention_score_mean": "attention_score",
}

# 行为事件类型（与 docs/03-数据库/数据库设计.md 一致）
BEHAVIOR_TYPES: List[str] = [
    "blink",
    "yawn",
    "head_down",
    "looking_away",
    "absent",
]

__all__ = [
    "STATE_FEATURE_COLUMNS",
    "AGGREGATION_SUFFIXES",
    "STATISTICAL_FEATURE_COLUMNS",
    "NUMERIC_STATE_COLUMNS",
    "BEHAVIOR_TYPES",
    "BEHAVIOR_TYPE_COLUMNS",
    "BEHAVIOR_STAT_COLUMNS",
    "BEHAVIOR_FEATURE_COLUMNS",
    "FINAL_STATISTICAL_COLUMNS",
    "FEATURE_VECTOR_COLUMNS",
    "FEATURE_VECTOR_STAT_COLUMNS",
    "FEATURE_VECTOR_RAW_COLUMNS",
    "WINDOW_STAT_FUNCTIONS",
    "STAT_TO_FEATURE_MAPPING",
]
