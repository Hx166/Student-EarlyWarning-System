"""src/models/features/builder.py —— 特征向量构建与导出。

从数据库读取的 StateRecord / BehaviorRecord 记录构建机器学习特征：
- 按滑动时间窗口计算统计特征；
- 统计行为事件次数与累计时长；
- 合并生成 Feature Vector；
- 按 Min-Max 归一化；
- 导出 CSV 供 XGBoost/LSTM 等模型读取。

数据来源以 StateRecord 为主，行为类特征按需读取 BehaviorRecord，
不修改数据库 Schema。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.models.features.definitions import (
    FEATURE_VECTOR_COLUMNS,
    NUMERIC_STATE_COLUMNS,
    STATISTICAL_FEATURE_COLUMNS,
)
from src.models.features.descriptors import (
    aggregate_state_statistics,
    compute_behavior_statistics,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_WINDOW_SECONDS = 30

_BEHAVIOR_COLUMNS = [
    "blink_count",
    "blink_rate",
    "head_down_count",
    "head_down_duration",
    "face_lost_count",
    "face_lost_duration",
    "yawn_count",
]


class FeatureBuilder:
    """特征向量构建器。

    负责将数据库中的状态记录与行为记录转换为
    机器学习模型可直接使用的结构化特征向量。

    Attributes:
        window_seconds: 特征统计时间窗口（秒）。
    """

    def __init__(self, window_seconds: int = DEFAULT_WINDOW_SECONDS) -> None:
        """初始化特征构建器。

        Args:
            window_seconds: 特征统计时间窗口长度（秒）。
        """
        self.window_seconds = window_seconds
        logger.info("FeatureBuilder 已初始化，窗口长度=%s 秒", window_seconds)

    def build(
        self,
        state_records: List[Dict[str, Any]],
        behavior_records: Optional[List[Dict[str, Any]]] = None,
        labels: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """从状态与行为记录构建特征向量 DataFrame。

        Args:
            state_records: StateRecord 记录字典列表。
            behavior_records: BehaviorRecord 记录字典列表（可空）。
            labels: 每条状态记录对应的标签列表（可为空）。

        Returns:
            特征向量 DataFrame；状态记录为空时返回空 DataFrame。
        """
        if not state_records:
            logger.warning("状态记录为空，返回空特征 DataFrame")
            return pd.DataFrame(columns=FEATURE_VECTOR_COLUMNS)

        state_df = pd.DataFrame(state_records)
        for col in NUMERIC_STATE_COLUMNS:
            if col not in state_df.columns:
                raise ValueError(f"状态记录缺少特征列: {col}")

        stat_df = aggregate_state_statistics(state_df, self.window_seconds)
        stat_df = stat_df[STATISTICAL_FEATURE_COLUMNS + ["record_time"]].copy()

        behavior_df = pd.DataFrame(behavior_records) if behavior_records else pd.DataFrame()
        stat_df = self._merge_behavior_stats(stat_df, behavior_df)

        features = self._build_feature_vector(stat_df)
        if labels is not None and len(labels) == len(features):
            features["label"] = list(labels)
        return features

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """对特征向量进行 Min-Max 归一化。

        Args:
            df: 特征向量 DataFrame。

        Returns:
            归一化后的特征向量 DataFrame（保留非数值列）。
        """
        if df is None or df.empty:
            return df
        result = df.copy()
        numeric_cols = result.select_dtypes(include=["number"]).columns.tolist()
        for col in numeric_cols:
            col_min = result[col].min()
            col_max = result[col].max()
            if col_max - col_min > 0:
                result[col] = (result[col] - col_min) / (col_max - col_min)
            else:
                result[col] = 0.0
        return result

    def to_csv(
        self,
        df: pd.DataFrame,
        output_path: str,
        index: bool = False,
    ) -> str:
        """将特征向量 DataFrame 导出为 CSV 文件。

        Args:
            df: 特征向量 DataFrame。
            output_path: 输出文件路径。
            index: 是否写入行索引。

        Returns:
            实际写入的文件路径。
        """
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=index, encoding="utf-8")
        logger.info("特征 CSV 已导出: %s（%s 行）", output_path, len(df))
        return output_path

    def build_and_save(
        self,
        state_records: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        behavior_records: Optional[List[Dict[str, Any]]] = None,
        labels: Optional[Sequence[int]] = None,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """构建特征向量并可选导出 CSV。

        Args:
            state_records: 状态记录字典列表。
            output_path: CSV 输出路径（为空则不导出）。
            behavior_records: 行为记录字典列表（可空）。
            labels: 每条状态记录对应的标签列表（可为空）。
            normalize: 是否对特征进行归一化。

        Returns:
            构建（及可选归一化）后的特征向量 DataFrame。
        """
        features = self.build(state_records, behavior_records, labels)
        if normalize and not features.empty:
            features = self.normalize(features)
        if output_path:
            self.to_csv(features, output_path)
        return features

    def _merge_behavior_stats(
        self,
        stat_df: pd.DataFrame,
        behavior_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """合并行为事件统计特征到状态统计 DataFrame。

        Args:
            stat_df: 状态统计特征 DataFrame（含 record_time 列）。
            behavior_df: 行为事件 DataFrame（可为空）。

        Returns:
            合并了行为统计列的特征 DataFrame。
        """
        result = stat_df.copy()
        empty_stats = {col: 0.0 for col in _BEHAVIOR_COLUMNS}
        if behavior_df is None or behavior_df.empty or "start_time" not in behavior_df.columns:
            for col in _BEHAVIOR_COLUMNS:
                result[col] = 0.0
            return result

        behavior_start = pd.to_datetime(behavior_df["start_time"], errors="coerce")
        time_col = pd.to_datetime(stat_df["record_time"], errors="coerce")
        window_start = time_col - pd.to_timedelta(self.window_seconds, unit="s")

        behavior_stats = pd.DataFrame(index=stat_df.index, columns=_BEHAVIOR_COLUMNS, dtype=float)
        for idx, row_time in enumerate(time_col):
            if pd.isna(row_time):
                behavior_stats.loc[idx] = empty_stats
                continue
            mask = behavior_start.between(window_start.iloc[idx], row_time, inclusive="both")
            window_stats = compute_behavior_statistics(behavior_df.loc[mask])
            behavior_stats.loc[idx] = {col: window_stats[col] for col in _BEHAVIOR_COLUMNS}

        for col in _BEHAVIOR_COLUMNS:
            result[col] = behavior_stats[col].values
        return result

    def _build_feature_vector(self, stat_df: pd.DataFrame) -> pd.DataFrame:
        """从统计特征与行为特征构建最终 Feature Vector。

        Args:
            stat_df: 包含统计特征与行为统计列的特征 DataFrame。

        Returns:
            特征向量 DataFrame（列顺序与 FEATURE_VECTOR_COLUMNS 一致）。
        """
        desired = FEATURE_VECTOR_COLUMNS
        build_columns = [col for col in desired if col != "blink_rate"]
        for col in build_columns:
            if col not in stat_df.columns:
                stat_df[col] = 0.0
        vector = stat_df[build_columns].copy()
        blink_count = stat_df["blink_count"] if "blink_count" in stat_df.columns else 0.0
        vector["blink_rate"] = blink_count * 60.0 / self.window_seconds
        return vector[desired]
