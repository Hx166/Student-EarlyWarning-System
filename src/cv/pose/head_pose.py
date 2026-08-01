"""src/cv/pose/head_pose.py —— 头部姿态（Head Pose）估计模块。

负责估计头部俯仰角（Pitch）、偏航角（Yaw）与翻滚角（Roll）。
第一阶段提供默认返回值保证流程可运行，
真实算法（基于 Face Mesh 468 关键点）在第二阶段实现。
"""

from __future__ import annotations

import numpy as np

from src.core.models import HeadPoseResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HeadPoseEstimator:
    """头部姿态估计器。

    Attributes:
        head_down_threshold: 低头判定 Pitch 阈值。
    """

    def __init__(self) -> None:
        """初始化头部姿态估计器。"""
        self.head_down_threshold = 35.0
        logger.info("HeadPoseEstimator 已初始化")

    def estimate(self, landmarks=None, frame: np.ndarray = None) -> HeadPoseResult:
        """估计头部姿态角。

        Args:
            landmarks: 人脸关键点数据（第二阶段使用）。
            frame: 视频帧（备用）。

        Returns:
            头部姿态结果对象。
        """
        # TODO(阶段二): 基于 Face Mesh 关键点 + 3D 模型求解姿态
        # 第一阶段返回默认姿态（正视）
        return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, confidence=0.5)
