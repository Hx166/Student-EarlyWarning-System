"""阶段 G 页面与演示 API 路由测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.app import create_app
from src.api.routes import demo


class _FakeDemoService:
    """不执行视频解码的演示服务桩。"""

    def get_clips(self) -> list[dict]:
        """返回 20 条假清单。"""
        return [
            {
                "clip_id": f"clip{i}.mp4" if i == 0 else f"clip{i}.avi",
                "engagement": i % 4,
                "frames": 300,
                "ext": ".mp4" if i == 0 else ".avi",
                "playable": i == 0,
            }
            for i in range(20)
        ]

    def run(self, clip_id: str) -> dict:
        """返回固定回放结构。"""
        if clip_id != "clip0.mp4":
            raise KeyError(clip_id)
        return {
            "clip_id": clip_id,
            "true_level": "level_0",
            "pred_level": "level_2",
            "confidence": 0.72,
            "probabilities": {
                "level_0": 0.05,
                "level_1": 0.08,
                "level_2": 0.72,
                "level_3": 0.15,
            },
            "series": [{"second": 1, "attention": 80, "fatigue": 12}],
            "elapsed_seconds": 0.1,
        }

    def video_path(self, clip_id: str) -> Path:
        """模拟受控视频路径。"""
        raise FileNotFoundError(clip_id)


def test_pages_and_demo_clips() -> None:
    """三个页面可访问，回放清单返回 20 条。"""
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/playback").status_code == 200
    assert client.get("/model").status_code == 200
    sample = Path(
        r"D:\Student-EarlyWarning-System\datasets\processed\stage_c_sample.csv"
    )
    if not sample.exists():
        pytest.skip("阶段 C 演示清单不存在")
    demo.reset_demo_service()
    response = client.get("/api/demo/clips")
    assert response.status_code == 200
    assert len(response.get_json()["data"]) == 20


def test_demo_run_structure_with_fake_service(monkeypatch) -> None:
    """回放接口应返回曲线、等级和概率结构。"""
    monkeypatch.setattr(demo, "get_demo_service", lambda: _FakeDemoService())
    app = create_app()
    app.config["TESTING"] = True
    response = app.test_client().get("/api/demo/run/clip0.mp4")
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["true_level"] == "level_0"
    assert data["pred_level"] == "level_2"
    assert len(data["probabilities"]) == 4
    assert data["series"]


def test_demo_video_rejects_unknown_and_illegal_paths(monkeypatch) -> None:
    """未知片段和路径穿越请求均返回 404。"""
    monkeypatch.setattr(demo, "get_demo_service", lambda: _FakeDemoService())
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    assert client.get("/api/demo/video/unknown.mp4").status_code == 404
    assert client.get("/api/demo/video/..%2Fsecret.mp4").status_code == 404
