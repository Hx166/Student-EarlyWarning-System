"""src/cv/eye/eye_ear.py —— 眼部状态（EyeEAR）检测模块。

通过眼部特征计算眼睛纵横比（EAR）并判断眼睛开闭状态。
第一阶段提供默认返回值保证流程可运行，
真实算法（MediaPipe 468 关键点）在第二阶段实现。
"""

from __future__ import annotations

import numpy as np

from src.config import get_camera
from src.core.models import EyeState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 左眼 MediaPipe 关键点编号（EyeEAR 设计文档定义）
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
# 右眼 MediaPipe 关键点编号
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]


class EyeEarProcessor:
    """眼部纵横比（EyeEAR）处理器。

    Attributes:
        ear_threshold: 闭眼判定阈值。
    """

    def __init__(self) -> None:
        """初始化眼部状态处理器。"""
        self.ear_threshold = 0.20
        logger.info("EyeEarProcessor 已初始化，阈值=%s", self.ear_threshold)

    def compute_ear(self, eye_landmarks) -> float:
        """计算单只眼睛的纵横比（EAR）。

        Args:
            eye_landmarks: 眼睛的 6 个关键点坐标列表。

        Returns:
            EAR 值；关键点不足时返回 0.0。
        """
        if eye_landmarks is None or len(eye_landmarks) < 6:
            return 0.0

        p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
        vertical_a = np.linalg.norm(p2 - p6)
        vertical_b = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal < 1e-6:
            return 0.0
        return float((vertical_a + vertical_b) / (2.0 * horizontal))

    def analyze(self, landmarks=None, frame: np.ndarray = None) -> EyeState:
        """分析眼部状态。

        Args:
            landmarks: 人脸关键点数据（第二阶段使用）。
            frame: 视频帧（备用）。

        Returns:
            眼部状态结果对象。
        """
        # TODO(阶段二): 基于 Face Mesh 关键点计算真实 EAR
        # 第一阶段返回默认睁眼状态
        return EyeState(
            left_ear=0.35,
            right_ear=0.35,
            average_ear=0.35,
            eye_state="open",
            blink=False,
            perclos=0.0,
            confidence=0.5,
        )
