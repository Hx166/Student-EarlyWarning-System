"""src/models/features/descriptors.py —— 特征统计描述。

提供对连续状态特征的行为事件列进行统计描述的函数。
每个特征按 30 秒滑动窗口计算 mean/std/min/max/median，
行为事件按窗口统计次数与累计时长，供特征构建模块使用。
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.models.features.definitions import (
    NUMERIC_STATE_COLUMNS,
    STATISTICAL_FEATURE_COLUMNS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_statistical_features(
    window_df: pd.DataFrame,
) -> Dict[str, float]:
    """计算单个时间窗口的统计特征描述。

    Args:
        window_df: 单个窗口内的状态记录 DataFrame。

    Returns:
        统计特征字典（mean/std/min/max/median）。
    """
    features: Dict[str, float] = {}
    for col in NUMERIC_STATE_COLUMNS:
        series = pd.to_numeric(window_df[col], errors="coerce")
        features[f"{col}_mean"] = float(series.mean() if series.notna().any() else 0.0)
        features[f"{col}_std"] = float(series.std() if series.notna().sum() > 1 else 0.0)
        features[f"{col}_min"] = float(series.min() if series.notna().any() else 0.0)
        features[f"{col}_max"] = float(series.max() if series.notna().any() else 0.0)
        features[f"{col}_median"] = float(series.median() if series.notna().any() else 0.0)
    return features


def compute_behavior_statistics(
    behavior_df: pd.DataFrame,
) -> Dict[str, float]:
    """计算行为事件统计特征。

    Args:
        behavior_df: 时间窗口内的行为事件 DataFrame。

    Returns:
        行为统计特征字典（次数与累计时长）。
    """
    stats: Dict[str, float] = {
        "blink_count": 0.0,
        "blink_rate": 0.0,
        "head_down_count": 0.0,
        "head_down_duration": 0.0,
        "face_lost_count": 0.0,
        "face_lost_duration": 0.0,
        "yawn_count": 0.0,
    }
    if behavior_df is None or behavior_df.empty:
        return stats

    for behavior_type, count_key in (
        ("blink", "blink_count"),
        ("head_down", "head_down_count"),
        ("looking_away", "face_lost_count"),
        ("absent", "face_lost_count"),
        ("yawn", "yawn_count"),
    ):
        mask = behavior_df["behavior_type"] == behavior_type
        count = int(mask.sum())
        stats[count_key] += float(count)
        duration = float(pd.to_numeric(behavior_df.loc[mask, "duration"], errors="coerce").sum())
        if behavior_type == "head_down":
            stats["head_down_duration"] += duration
        elif behavior_type in ("looking_away", "absent"):
            stats["face_lost_duration"] += duration
    return stats


def aggregate_state_statistics(
    df: pd.DataFrame,
    window_seconds: int = 30,
) -> pd.DataFrame:
    """按滑动时间窗口聚合状态记录的统计特征。

    Args:
        df: 包含 record_time 与数值状态列的状态记录 DataFrame。
        window_seconds: 滑动窗口长度（秒），默认 30。

    Returns:
        聚合后的统计特征 DataFrame（含 record_time 列，行按时间升序）。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    time_col = "record_time"
    if time_col not in df.columns:
        raise ValueError("状态记录缺少 record_time 列")
    for col in NUMERIC_STATE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"状态记录缺少数值列: {col}")

    df = df.sort_values(time_col).copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    agg_functions = {
        col: ["mean", "std", "min", "max", "median"]
        for col in NUMERIC_STATE_COLUMNS
    }
    rolled = (
        df[NUMERIC_STATE_COLUMNS]
        .rolling(f"{window_seconds}s", min_periods=1)
        .agg(agg_functions)
    )
    rolled.columns = STATISTICAL_FEATURE_COLUMNS
    rolled[time_col] = df.index
    return rolled.reset_index(drop=True)
