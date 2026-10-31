"""Engagement 独立标签映射测试。"""

from __future__ import annotations

from src.models.labels import ENGAGEMENT_LABEL_MAPPING, ENGAGEMENT_LABEL_NAMES
from src.models.xgboost.model import LABEL_MAPPING


def test_engagement_mapping_has_level_zero_to_three() -> None:
    """Engagement ID 和名称应严格对应 0~3。"""
    assert ENGAGEMENT_LABEL_MAPPING == {
        0: "level_0",
        1: "level_1",
        2: "level_2",
        3: "level_3",
    }
    assert ENGAGEMENT_LABEL_NAMES == [
        "level_0",
        "level_1",
        "level_2",
        "level_3",
    ]


def test_engagement_mapping_is_distinct_from_risk_mapping() -> None:
    """Engagement 名称不得与风险状态名称混用。"""
    assert ENGAGEMENT_LABEL_MAPPING != LABEL_MAPPING
    assert not set(ENGAGEMENT_LABEL_NAMES).intersection(LABEL_MAPPING.values())
