"""阶段 C 聚合与固定抽样逻辑测试（不依赖 cv2/mediapipe）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.stage_c_features import (
    FEATURE_VECTOR_COLUMNS,
    independent_features,
    production_features,
    sample_clips,
)


def _synthetic_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 10 条 state 与少量行为事件。"""
    rows = []
    for index in range(10):
        rows.append(
            {
                "record_time": f"2026-01-01T00:00:{index + 1:02d}",
                "avg_ear": 0.25 + index * 0.01,
                "avg_mar": 0.10 + index * 0.005,
                "avg_pitch": -2.0 + index,
                "avg_yaw": 3.0 - index * 0.5,
                "avg_roll": 1.0 + index * 0.2,
                "perclos": index * 0.01,
                "attention_score": 80.0 - index,
                "fatigue_score": 10.0 + index,
            }
        )
    behavior = pd.DataFrame(
        [
            {"behavior_type": "blink", "start_time": "2026-01-01T00:00:03", "duration": 0.0},
            {"behavior_type": "blink", "start_time": "2026-01-01T00:00:07", "duration": 0.0},
            {"behavior_type": "head_down", "start_time": "2026-01-01T00:00:02", "duration": 3.0},
            {"behavior_type": "looking_away", "start_time": "2026-01-01T00:00:08", "duration": 1.5},
        ]
    )
    return pd.DataFrame(rows), behavior


def test_production_and_independent_aggregation_match() -> None:
    """10 条记录恰好构成完整窗口，A/B 11 列应一致。"""
    state, behavior = _synthetic_records()
    production = production_features(state, behavior)
    independent = independent_features(state, behavior)
    for feature in FEATURE_VECTOR_COLUMNS:
        assert abs(float(production[feature]) - float(independent[feature])) <= 1e-6


def test_sampling_is_stable_and_forces_two_anomaly_mp4(tmp_path: Path) -> None:
    """同 seed 抽样稳定，并至少包含两条 302/314 帧 MP4。"""
    rows = []
    for engagement in range(4):
        for index in range(8):
            clip_id = f"{engagement + 1:02d}{index:03d}.avi"
            rows.append(
                {
                    "clip_id": clip_id,
                    "subject_id": f"S{engagement}",
                    "split": "Train",
                    "engagement": engagement,
                    "frames": 300,
                    "ext": ".avi",
                    "path": str(tmp_path / clip_id),
                }
            )
    rows.extend(
        [
            {
                "clip_id": "901001.mp4", "subject_id": "S2", "split": "Train",
                "engagement": 2, "frames": 302, "ext": ".mp4", "path": str(tmp_path / "901001.mp4"),
            },
            {
                "clip_id": "902001.mp4", "subject_id": "S3", "split": "Train",
                "engagement": 3, "frames": 314, "ext": ".mp4", "path": str(tmp_path / "902001.mp4"),
            },
        ]
    )
    manifest = pd.DataFrame(rows)
    first, _ = sample_clips(manifest, seed=42)
    second, _ = sample_clips(manifest, seed=42)
    assert first["clip_id"].tolist() == second["clip_id"].tolist()
    assert len(first) == 20
    assert first.groupby("engagement").size().eq(5).all()
    anomaly = first[(first["ext"] == ".mp4") & first["frames"].isin([302, 314])]
    assert len(anomaly) >= 2
