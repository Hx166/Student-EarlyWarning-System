"""src/cv/processor.py —— 计算机视觉处理门面（CVProcessor）。

将人脸检测、眼部状态、头部姿态、疲劳与参与度分析等
子模块组合为统一的视觉处理流水线。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.core.models import (
    EngagementResult,
    EyeState,
    FaceLandmark,
    FatigueResult,
    HeadPoseResult,
)
from src.cv.face.detector import FaceDetector
from src.cv.eye.eye_ear import EyeEarProcessor
from src.cv.pose.head_pose import HeadPoseEstimator
from src.cv.behavior.fatigue import FatigueAnalyzer
from src.cv.emotion.engagement import EngagementAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CVProcessResult:
    """单帧视觉处理结果聚合对象。

    Attributes:
        faces: 检测到的人脸列表。
        eye: 眼部状态结果。
        head_pose: 头部姿态结果。
        fatigue: 疲劳评估结果。
        engagement: 课堂参与度结果。
    """

    faces: List[FaceLandmark] = None
    eye: Optional[EyeState] = None
    head_pose: Optional[HeadPoseResult] = None
    fatigue: Optional[FatigueResult] = None
    engagement: Optional[EngagementResult] = None

    def __post_init__(self) -> None:
        """默认空结果初始化。"""
        if self.faces is None:
            self.faces = []
        if self.eye is None:
            self.eye = EyeState()
        if self.head_pose is None:
            self.head_pose = HeadPoseResult()
        if self.fatigue is None:
            self.fatigue = FatigueResult()
        if self.engagement is None:
            self.engagement = EngagementResult()


class CVProcessor:
    """视觉处理流水线门面。

    统一管理人脸检测、眼部状态、头部姿态、疲劳与参与度分析。
    对外仅暴露 process 接口。

    Attributes:
        face_detector: 人脸检测器。
        eye_processor: 眼部状态处理器。
        head_pose_estimator: 头部姿态估计器。
        fatigue_analyzer: 疲劳评估器。
        engagement_analyzer: 参与度评估器。
    """

    def __init__(self) -> None:
        """初始化视觉处理流水线及各子模块。"""
        self.face_detector = FaceDetector()
        self.eye_processor = EyeEarProcessor()
        self.head_pose_estimator = HeadPoseEstimator()
        self.fatigue_analyzer = FatigueAnalyzer()
        self.engagement_analyzer = EngagementAnalyzer()
        logger.info("CVProcessor 初始化完成")

    def process(self, frame: np.ndarray) -> CVProcessResult:
        """处理单帧图像并返回完整视觉分析结果。

        Args:
            frame: OpenCV BGR 图像矩阵。

        Returns:
            聚合的视觉处理结果对象。
        """
        faces = self.face_detector.detect(frame)
        eye = self.eye_processor.analyze(frame=frame)
        head_pose = self.head_pose_estimator.estimate(frame=frame)
        fatigue = self.fatigue_analyzer.analyze(
            eye_state=eye, head_pose=head_pose
        )
        engagement = self.engagement_analyzer.analyze(
            eye_state=eye, head_pose=head_pose, fatigue=fatigue
        )
        return CVProcessResult(
            faces=faces,
            eye=eye,
            head_pose=head_pose,
            fatigue=fatigue,
            engagement=engagement,
        )

    def close(self) -> None:
        """释放所有视觉子模块资源。"""
        self.face_detector.close()
        logger.info("CVProcessor 已关闭")
