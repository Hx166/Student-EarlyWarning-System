"""src/cv/mouth/mouth_mar.py —— 嘴部状态（MAR）检测模块。

通过嘴部关键点计算嘴部纵横比（MAR）用于判断打哈欠等行为。
第一阶段提供默认返回值保证流程可运行。
"""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MouthProcessor:
    """嘴部状态处理器。

    Attributes:
        mar_threshold: 哈欠判定阈值。
    """

    def __init__(self) -> None:
        """初始化嘴部状态处理器。"""
        self.mar_threshold = 0.6
        logger.info("MouthProcessor 已初始化")

    def compute_mar(self, mouth_landmarks) -> float:
        """计算嘴部纵横比（MAR）。

        Args:
            mouth_landmarks: 嘴部关键点坐标列表。

        Returns:
            MAR 值；关键点不足时返回 0.0。
        """
        if mouth_landmarks is None or len(mouth_landmarks) < 6:
            return 0.0
        # TODO(阶段二): 基于嘴部关键点计算真实 MAR
        return 0.0
