"""最新 Engagement 预测接口测试。"""

from __future__ import annotations

from src.api.app import create_app


class _FakePipeline:
    """提供 last_prediction 的流水线桩。"""

    def __init__(self, prediction: dict | None) -> None:
        self.last_prediction = prediction


def test_prediction_latest_returns_pipeline_snapshot() -> None:
    """最新预测接口应返回假 Pipeline 的快照。"""
    expected = {
        "record_time": "2026-10-31T10:00:10",
        "model_name": "xgboost",
        "model_version": "engagement-v1",
        "level": "level_2",
        "probability": 0.75,
        "probabilities": {
            "level_0": 0.05,
            "level_1": 0.10,
            "level_2": 0.75,
            "level_3": 0.10,
        },
    }
    app = create_app(pipeline=_FakePipeline(expected))
    app.config["TESTING"] = True
    response = app.test_client().get("/api/prediction/latest")
    assert response.status_code == 200
    assert response.get_json()["data"] == expected


def test_prediction_latest_returns_null_without_runtime_prediction() -> None:
    """未接入摄像头时应返回 null 和 HTTP 200。"""
    app = create_app(pipeline=_FakePipeline(None))
    app.config["TESTING"] = True
    response = app.test_client().get("/api/prediction/latest")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["code"] == 0
    assert payload["data"] is None
