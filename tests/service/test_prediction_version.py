"""Engagement 模型版本与在线特征一致性测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.features.builder import FeatureBuilder
from src.models.registry import ModelRegistry
from src.service.prediction import PredictionService


def _state_records() -> list[dict]:
    """构造 10 条连续秒级状态记录。"""
    return [
        {
            "record_time": f"2026-10-31T10:00:{index + 1:02d}",
            "avg_ear": 0.30 + index * 0.002,
            "avg_mar": 0.10,
            "avg_pitch": 2.0 + index * 0.1,
            "avg_yaw": -1.0,
            "avg_roll": 0.5,
            "perclos": 0.05,
            "attention_score": 85.0,
            "fatigue_score": 15.0,
        }
        for index in range(10)
    ]


def test_prediction_service_passes_model_version() -> None:
    """PredictionService 应把版本号传给 RiskPredictor。"""
    service = PredictionService(model_version="engagement-v1")
    assert service._ensure_predictor().model_version == "engagement-v1"


def test_buffer_features_match_feature_builder(tmp_path: Path) -> None:
    """在线 buffer 构建的 11 列应与 FeatureBuilder(10) 一致。"""
    records = _state_records()
    service = PredictionService(
        window_seconds=10,
        min_records=10,
        registry=ModelRegistry(models_root=str(tmp_path / "models")),
        model_version="engagement-v1",
    )
    for record in records:
        service.predict(record)
    online = service._builder.build(list(service._buffer)).iloc[-1]
    training = FeatureBuilder(window_seconds=10).build(records).iloc[-1]
    np.testing.assert_allclose(online.to_numpy(), training.to_numpy(), atol=1e-12)


def test_behavior_records_are_included_in_online_features(tmp_path: Path) -> None:
    """在线构建应接收行为事件并生成对应行为特征。"""
    records = _state_records()
    behavior = [
        {
            "behavior_type": "blink",
            "start_time": "2026-10-31T10:00:03",
            "end_time": "2026-10-31T10:00:03",
            "duration": 0.0,
        },
        {
            "behavior_type": "looking_away",
            "start_time": "2026-10-31T10:00:05",
            "end_time": "2026-10-31T10:00:06",
            "duration": 1.0,
        },
    ]
    service = PredictionService(
        window_seconds=10,
        min_records=10,
        registry=ModelRegistry(models_root=str(tmp_path / "models")),
        model_version="engagement-v1",
    )
    for record in records:
        service.predict(record, behavior_records=behavior)
    online = service._builder.build(
        list(service._buffer), behavior
    ).iloc[-1]
    training = FeatureBuilder(window_seconds=10).build(
        records, behavior
    ).iloc[-1]
    np.testing.assert_allclose(online.to_numpy(), training.to_numpy(), atol=1e-12)
    assert online["blink_rate"] == 6.0
    assert online["face_lost_duration"] == 1.0
