"""阶段 G 回放服务的清单、路径校验与缓存测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.service.demo_playback import DemoPlaybackService


def _clips(tmp_path: Path) -> pd.DataFrame:
    """构造测试片段清单。"""
    return pd.DataFrame(
        [
            {
                "clip_id": "clip1.mp4",
                "engagement": 2,
                "frames": 300,
                "ext": ".mp4",
                "path": str(tmp_path / "clip1.mp4"),
            },
            {
                "clip_id": "clip2.avi",
                "engagement": 3,
                "frames": 302,
                "ext": ".avi",
                "path": str(tmp_path / "clip2.avi"),
            },
        ]
    )


def test_clip_list_and_invalid_id(tmp_path: Path) -> None:
    """清单字段正确，未知 clip_id 会被拒绝。"""
    service = DemoPlaybackService(
        sample_path=tmp_path / "missing.csv",
        model_version="engagement-v1",
        clips=_clips(tmp_path),
    )
    clips = service.get_clips()
    assert len(clips) == 2
    assert clips[0] == {
        "clip_id": "clip1.mp4",
        "engagement": 2,
        "frames": 300,
        "ext": ".mp4",
        "playable": True,
    }
    with pytest.raises(KeyError):
        service.run("../unknown.mp4")


def test_run_uses_cache_on_second_call(tmp_path: Path, monkeypatch) -> None:
    """同一片段第二次运行应直接命中缓存。"""
    service = DemoPlaybackService(
        sample_path=tmp_path / "missing.csv",
        model_version="engagement-v1",
        clips=_clips(tmp_path),
    )
    expected = {
        "clip_id": "clip1.mp4",
        "true_level": "level_2",
        "pred_level": "level_2",
        "confidence": 0.7,
        "probabilities": {
            "level_0": 0.05,
            "level_1": 0.1,
            "level_2": 0.7,
            "level_3": 0.15,
        },
        "series": [{"second": 1, "attention": 80.0, "fatigue": 10.0}],
        "elapsed_seconds": 0.01,
    }
    calls = []

    def fake_run(row: pd.Series) -> dict:
        calls.append(row["clip_id"])
        return expected

    monkeypatch.setattr(service, "_run_clip", fake_run)
    first = service.run("clip1.mp4")
    second = service.run("clip1.mp4")
    assert first == expected
    assert second == expected
    assert calls == ["clip1.mp4"]


def test_video_path_rejects_non_mp4_and_outside_root(tmp_path: Path) -> None:
    """视频文件只允许清单内 MP4 且必须位于 DAiSEE 根目录。"""
    service = DemoPlaybackService(
        sample_path=tmp_path / "missing.csv",
        model_version="engagement-v1",
        clips=_clips(tmp_path),
    )
    with pytest.raises(FileNotFoundError):
        service.video_path("clip2.avi")
    with pytest.raises(FileNotFoundError):
        service.video_path("clip1.mp4")
