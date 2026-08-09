"""tests/model/test_feature_engineering.py —— 特征工程模块测试。

覆盖统计描述计算、特征向量构建、行为统计合并、
Min-Max 归一化与 CSV 导出。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.models.features.builder import FeatureBuilder
from src.models.features.definitions import (
    FEATURE_VECTOR_COLUMNS,
    STATISTICAL_FEATURE_COLUMNS,
)
from src.models.features.descriptors import (
    compute_behavior_statistics,
    compute_statistical_features,
)


def _make_state_records(count: int = 35, base: int = 100) -> List[Dict[str, Any]]:
    """生成用于测试的状态记录。

    Args:
        count: 记录条数。
        base: 起始毫秒偏移。

    Returns:
        状态记录字典列表。
    """
    start = datetime(2026, 8, 9, 9, 0, 0)
    records: List[Dict[str, Any]] = []
    for i in range(count):
        records.append(
            {
                "state_id": base + i,
                "student_id": 1,
                "session_id": 1,
                "record_time": (start + timedelta(seconds=i)).isoformat(),
                "avg_ear": 0.30 + i * 0.001,
                "avg_mar": 0.10,
                "avg_pitch": 5.0 + i * 0.1,
                "avg_yaw": -2.0,
                "avg_roll": 1.0,
                "perclos": 0.12,
                "attention_score": 80.0 - i * 0.1,
                "fatigue_score": 20.0 + i * 0.1,
            }
        )
    return records


def _make_behavior_records(count: int = 5) -> List[Dict[str, Any]]:
    """生成用于测试的行为事件记录。

    Args:
        count: 事件条数。

    Returns:
        行为事件记录字典列表。
    """
    start = datetime(2026, 8, 9, 9, 0, 2)
    records: List[Dict[str, Any]] = []
    types = ["blink", "blink", "head_down", "looking_away", "yawn"]
    for i in range(count):
        records.append(
            {
                "behavior_id": i + 1,
                "student_id": 1,
                "session_id": 1,
                "behavior_type": types[i % len(types)],
                "start_time": (start + timedelta(seconds=i * 3)).isoformat(),
                "end_time": (start + timedelta(seconds=i * 3 + 1)).isoformat(),
                "duration": 1.0,
            }
        )
    return records


class FeatureBuilderTest(unittest.TestCase):
    """特征构建器测试。"""

    def setUp(self) -> None:
        """初始化测试数据与构建器。"""
        self.state_records = _make_state_records()
        self.behavior_records = _make_behavior_records()
        self.builder = FeatureBuilder(window_seconds=30)

    def test_build_feature_vector_columns(self):
        """构建结果应包含标准特征向量列。"""
        features = self.builder.build(self.state_records, self.behavior_records)
        self.assertEqual(list(features.columns), FEATURE_VECTOR_COLUMNS)
        self.assertEqual(len(features), 35)

    def test_build_with_behaviors_positive(self):
        """存在行为记录时行为特征应出现正值。"""
        features = self.builder.build(self.state_records, self.behavior_records)
        self.assertGreater(float(features["head_down_duration"].sum()), 0.0)
        self.assertGreater(float(features["face_lost_duration"].sum()), 0.0)
        self.assertGreater(float(features["blink_rate"].sum()), 0.0)
    def test_build_without_behavior(self):
        """无行为记录时行为特征应为 0。"""
        features = self.builder.build(self.state_records)
        self.assertAlmostEqual(float(features["head_down_duration"].sum()), 0.0, places=6)
        self.assertAlmostEqual(float(features["blink_rate"].sum()), 0.0, places=6)
        self.assertAlmostEqual(float(features["face_lost_duration"].sum()), 0.0, places=6)

    def test_build_empty_state_returns_empty(self):
        """空状态记录应返回空 DataFrame。"""
        features = self.builder.build([])
        self.assertTrue(features.empty)
        self.assertEqual(list(features.columns), FEATURE_VECTOR_COLUMNS)

    def test_normalize_limits(self):
        """归一化后数值特征应在 0~1 之间。"""
        features = self.builder.build(self.state_records, self.behavior_records)
        normalized = self.builder.normalize(features)
        numeric = normalized.select_dtypes(include=["number"])
        self.assertGreaterEqual(float(numeric.min().min()), 0.0)
        self.assertLessEqual(float(numeric.max().max()), 1.0)

    def test_to_csv(self):
        """导出 CSV 文件应存在且可读取。"""
        features = self.builder.build(self.state_records)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "features.csv")
            written = self.builder.to_csv(features, path)
            self.assertTrue(os.path.exists(written))
            with open(written, "r", encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn("ear_mean", content)

    def test_labels_appended(self):
        """标签应与特征行数一致并追加。"""
        labels = [1] * len(self.state_records)
        features = self.builder.build(self.state_records, labels=labels)
        self.assertIn("label", features.columns)
        self.assertEqual(int(features["label"].iloc[0]), 1)

    def test_statistical_columns_present(self):
        """特征向量应包含文档要求的统计特征列。"""
        features = self.builder.build(self.state_records, self.behavior_records)
        required_stat = ["ear_mean", "ear_std", "pitch_mean", "yaw_mean", "roll_mean"]
        for col in required_stat:
            self.assertIn(col, features.columns)


class StatisticalFunctionsTest(unittest.TestCase):
    """统计描述函数测试。"""

    def test_compute_statistical_features(self):
        """统计特征应包含 mean/std/min/max/median。"""
        import pandas as pd

        window = pd.DataFrame(
            [
                {"avg_ear": 0.3, "avg_mar": 0.1, "avg_pitch": 1.0,
                 "avg_yaw": 2.0, "avg_roll": 3.0, "perclos": 0.1,
                 "attention_score": 80.0, "fatigue_score": 20.0},
                {"avg_ear": 0.5, "avg_mar": 0.2, "avg_pitch": 3.0,
                 "avg_yaw": 4.0, "avg_roll": 5.0, "perclos": 0.3,
                 "attention_score": 60.0, "fatigue_score": 40.0},
            ]
        )
        features = compute_statistical_features(window)
        self.assertAlmostEqual(features["avg_ear_mean"], 0.4, places=6)
        self.assertAlmostEqual(features["avg_ear_min"], 0.3, places=6)
        self.assertAlmostEqual(features["avg_ear_max"], 0.5, places=6)
        self.assertAlmostEqual(features["avg_ear_median"], 0.4, places=6)
        self.assertGreater(features["avg_ear_std"], 0.0)

    def test_compute_behavior_statistics(self):
        """行为统计应正确累计次数与时长。"""
        import pandas as pd

        behaviors = pd.DataFrame(
            [
                {"behavior_type": "blink", "duration": 1.0},
                {"behavior_type": "blink", "duration": 2.0},
                {"behavior_type": "head_down", "duration": 3.0},
                {"behavior_type": "looking_away", "duration": 4.0},
                {"behavior_type": "absent", "duration": 5.0},
                {"behavior_type": "yawn", "duration": 6.0},
            ]
        )
        stats = compute_behavior_statistics(behaviors)
        self.assertEqual(stats["blink_count"], 2.0)
        self.assertEqual(stats["head_down_count"], 1.0)
        self.assertEqual(stats["head_down_duration"], 3.0)
        self.assertEqual(stats["face_lost_count"], 2.0)
        self.assertEqual(stats["face_lost_duration"], 9.0)
        self.assertEqual(stats["yawn_count"], 1.0)
        self.assertAlmostEqual(stats["blink_rate"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
