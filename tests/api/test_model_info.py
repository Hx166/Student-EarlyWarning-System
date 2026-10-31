"""Engagement 模型信息接口测试。"""

from __future__ import annotations

from src.api.app import create_app
from src.api.service import prediction as prediction_service


def test_model_info_returns_engagement_version_and_mapping() -> None:
    """模型信息接口应返回 engagement-v1 与独立标签映射。"""
    prediction_service.reset()
    app = create_app()
    app.config["TESTING"] = True
    payload = app.test_client().get("/api/model/info").get_json()
    assert prediction_service.get_model_version() == "engagement-v1"
    assert payload["code"] == 0
    assert payload["data"]["version"] == "engagement-v1"
    assert len(payload["data"]["feature_names"]) == 11
    assert payload["data"]["label_mapping"] == {
        "0": "level_0",
        "1": "level_1",
        "2": "level_2",
        "3": "level_3",
    }
