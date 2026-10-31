"""DAiSEE Engagement 四分类标签定义。

本映射独立于风险预测使用的 ``LABEL_MAPPING``，禁止混用：
Engagement 0 表示最低参与度，3 表示最高参与度。
"""

from __future__ import annotations

from typing import Dict, List

ENGAGEMENT_LABEL_MAPPING: Dict[int, str] = {
    0: "level_0",
    1: "level_1",
    2: "level_2",
    3: "level_3",
}

ENGAGEMENT_LABEL_NAMES: List[str] = [
    ENGAGEMENT_LABEL_MAPPING[index] for index in range(4)
]

__all__ = ["ENGAGEMENT_LABEL_MAPPING", "ENGAGEMENT_LABEL_NAMES"]
