"""src/cv/mouth/mouth_mar.py —— 嘴部状态（MAR）检测模块。

基于 MediaPipe Face Mesh 关键点计算嘴部纵横比（MAR），
用于判断打哈欠等嘴部行为。计算思路参考 EyeEAR 的 EAR 公式。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.core.models import FaceLandmark, MouthState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 嘴部 MediaPipe 关键点编号（顺序与 EAR 公式布局一致）：
# [左嘴角, 上唇外, 上唇内, 右嘴角, 下唇内, 下唇外]
MOUTH_LANDMARKS = [61, 13, 0, 291, 17, 14]


class MouthProcessor:
    """嘴部状态处理器。

    依据嘴部 6 个关键点计算 MAR，并基于阈值判定打哈欠。

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
            mouth_landmarks: 嘴部 6 个关键点坐标数组。

        Returns:
            MAR 值；关键点不足时返回 0.0。
        """
        if mouth_landmarks is None or len(mouth_landmarks) < 6:
            return 0.0

        p1, p2, p3, p4, p5, p6 = mouth_landmarks[:6]
        vertical_a = np.linalg.norm(p2 - p6)
        vertical_b = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-6:
            return 0.0
        return float((vertical_a + vertical_b) / (2.0 * horizontal))

    def analyze(
        self, landmarks: Optional[FaceLandmark] = None
    ) -> MouthState:
        """分析嘴部状态并判定打哈欠。

        Args:
            landmarks: 人脸关键点对象（FaceLandmark）。

        Returns:
            嘴部状态结果对象；输入不合法时返回默认状态。
        """
        if not isinstance(landmarks, FaceLandmark):
            return MouthState()

        lms = landmarks.landmarks
        points = self._extract_points(lms)
        if points is None:
            return MouthState()

        mar = self.compute_mar(points)
        return MouthState(
            mar=mar,
            is_yawn=mar >= self.mar_threshold,
            confidence=1.0,
        )

    def _extract_points(self, landmarks) -> Optional[np.ndarray]:
        """提取嘴部关键点坐标数组。

        Args:
            landmarks: Face Landmark 关键点列表。

        Returns:
            嘴部关键点坐标数组；数据缺失或越界时返回 None。
        """
        if landmarks is None:
            return None
        points = []
        for idx in MOUTH_LANDMARKS:
            if idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            points.append([lm[0], lm[1], lm[2] if len(lm) > 2 else 0.0])
        return np.asarray(points, dtype=np.float64)
