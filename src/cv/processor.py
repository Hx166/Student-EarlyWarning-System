"""src/cv/processor.py —— 计算机视觉处理门面（CVProcessor）。

将人脸检测、眼部状态、头部姿态、疲劳与参与度分析等
子模块组合为统一的视觉处理流水线。

本模块负责创建唯一的 MediaPipeProcessor 实例，
并通过依赖注入共享给 FaceDetector、EyeEarProcessor、HeadPoseEstimator，
避免多个模块重复初始化模型引擎。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.models import (
    EngagementResult,
    EyeState,
    FaceAnalysis,
    FaceLandmark,
    FatigueResult,
    HeadPoseResult,
    MouthState,
    PoseLandmark,
)
from src.cv.mediapipe import MediaPipeProcessor
from src.cv.face.detector import FaceDetector
from src.cv.eye.eye_ear import EyeEarProcessor
from src.cv.pose.head_pose import HeadPoseEstimator
from src.cv.behavior.fatigue import FatigueAnalyzer
from src.cv.emotion.engagement import EngagementAnalyzer
from src.cv.mouth.mouth_mar import MouthProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CVProcessResult:
    """单帧视觉处理结果聚合对象。

    Attributes:
        faces: 检测到的人脸列表。
        eye: 眼部状态结果（主脸，向后兼容）。
        head_pose: 头部姿态结果（主脸，向后兼容）。
        mouth: 嘴部状态结果（主脸，向后兼容）。
        fatigue: 疲劳评估结果（主脸，向后兼容）。
        engagement: 课堂参与度结果（主脸，向后兼容）。
        pose: 人体姿态关键点（可为 None）。
        face_analyses: 全部人脸逐人分析结果列表。
    """

    faces: List[FaceLandmark] = None
    eye: Optional[EyeState] = None
    head_pose: Optional[HeadPoseResult] = None
    mouth: Optional[MouthState] = None
    fatigue: Optional[FatigueResult] = None
    engagement: Optional[EngagementResult] = None
    pose: Optional[PoseLandmark] = None
    face_analyses: List[FaceAnalysis] = None

    def __post_init__(self) -> None:
        """默认空结果初始化。"""
        if self.faces is None:
            self.faces = []
        if self.eye is None:
            self.eye = EyeState()
        if self.head_pose is None:
            self.head_pose = HeadPoseResult()
        if self.mouth is None:
            self.mouth = MouthState()
        if self.fatigue is None:
            self.fatigue = FatigueResult()
        if self.engagement is None:
            self.engagement = EngagementResult()
        if self.face_analyses is None:
            self.face_analyses = []


class CVProcessor:
    """视觉处理流水线门面。

    统一管理人脸检测、眼部状态、头部姿态、疲劳与参与度分析。
    对外暴露 process 接口，对多人场景逐人分析。

    Attributes:
        mediapipe: 共享的 MediaPipe 关键点检测处理器（单实例）。
        face_detector: 人脸检测器。
        eye_processor: 眼部状态处理器（主脸）。
        head_pose_estimator: 头部姿态估计器。
        mouth_processor: 嘴部状态处理器。
        fatigue_analyzer: 疲劳评估器（主脸）。
        engagement_analyzer: 参与度评估器。
    """

    def __init__(self) -> None:
        """初始化视觉处理流水线及各子模块。"""
        # 创建唯一 MediaPipe 单实例并注入各视觉模块
        self.mediapipe = MediaPipeProcessor()
        self.face_detector = FaceDetector(mediapipe=self.mediapipe)
        self.eye_processor = EyeEarProcessor(mediapipe=self.mediapipe)
        self.head_pose_estimator = HeadPoseEstimator(mediapipe=self.mediapipe)
        self.mouth_processor = MouthProcessor()
        self.fatigue_analyzer = FatigueAnalyzer()
        self.engagement_analyzer = EngagementAnalyzer()
        # 其他人脸的状态型分析器（按 face_id 惰性创建，避免状态互相污染）
        self._extra_eye_processors: Dict[int, EyeEarProcessor] = {}
        self._extra_fatigue_analyzers: Dict[int, FatigueAnalyzer] = {}
        logger.info("CVProcessor 初始化完成，MediaPipe 单实例已共享")

    def process(self, frame: np.ndarray) -> CVProcessResult:
        """处理单帧图像并返回完整视觉分析结果。

        对检测到的每张人脸逐人分析，主脸结果保留在顶层字段（向后兼容）。

        Args:
            frame: OpenCV BGR 图像矩阵。

        Returns:
            聚合的视觉处理结果对象。
        """
        # 1. 人脸检测（MediaPipe Face Mesh）
        faces = self.face_detector.detect(frame)
        pose = self.face_detector.get_last_pose()

        # 2. 逐人分析
        analyses = [self._analyze_face(face, frame) for face in faces]

        # 3. 主脸结果保留在顶层字段，其余人脸通过 face_analyses 暴露
        if analyses:
            primary = analyses[0]
            return CVProcessResult(
                faces=faces,
                eye=primary.eye,
                head_pose=primary.head_pose,
                mouth=primary.mouth,
                fatigue=primary.fatigue,
                engagement=primary.engagement,
                pose=pose,
                face_analyses=analyses,
            )
        return CVProcessResult(faces=[], pose=pose, face_analyses=[])

    def _analyze_face(self, face: FaceLandmark, frame: np.ndarray) -> FaceAnalysis:
        """对单张人脸执行完整视觉分析。

        Args:
            face: 人脸关键点对象。
            frame: 原始帧（供头部姿态估计）。

        Returns:
            该人脸的完整分析结果。
        """
        eye_processor, fatigue_analyzer = self._face_analyzers(face.face_id)
        eye = eye_processor.analyze(landmarks=face, frame=frame)
        head_pose = self.head_pose_estimator.estimate(landmarks=face, frame=frame)
        mouth = self.mouth_processor.analyze(landmarks=face)
        fatigue = fatigue_analyzer.analyze(
            eye_state=eye, head_pose=head_pose, mouth_state=mouth
        )
        engagement = self.engagement_analyzer.analyze(
            eye_state=eye, head_pose=head_pose, fatigue=fatigue
        )
        return FaceAnalysis(
            face_id=face.face_id,
            eye=eye,
            head_pose=head_pose,
            mouth=mouth,
            fatigue=fatigue,
            engagement=engagement,
        )

    def _face_analyzers(
        self, face_id: int
    ) -> Tuple[EyeEarProcessor, FatigueAnalyzer]:
        """获取指定人脸的状态型分析器。

        Args:
            face_id: 人脸编号。

        Returns:
            (眼部状态处理器, 疲劳评估器) 元组。
        """
        if face_id == 0:
            return self.eye_processor, self.fatigue_analyzer
        eye_processor = self._extra_eye_processors.setdefault(
            face_id, EyeEarProcessor(mediapipe=self.mediapipe)
        )
        fatigue_analyzer = self._extra_fatigue_analyzers.setdefault(
            face_id, FatigueAnalyzer()
        )
        return eye_processor, fatigue_analyzer

    def close(self) -> None:
        """释放所有视觉子模块资源。"""
        # FaceDetector 为共享实例，关闭时由其统一释放 MediaPipe
        self.face_detector.close()
        self.mediapipe.close()
        logger.info("CVProcessor 已关闭")
