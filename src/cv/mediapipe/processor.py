"""src/cv/mediapipe/processor.py —— MediaPipe 关键点检测处理器。

封装 MediaPipe Face Mesh 与 Pose 两个解决方案，
将每帧图像检测并转换为项目统一的数据结构
（FaceLandmark / PoseLandmark），供后续 EyeEAR、HeadPose 等模块使用。

接口保持稳定，不随 MediaPipe 具体 API 变化而变化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.config import get_camera
from src.core.models import FaceLandmark, PoseLandmark
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:  # MediaPipe 为可选依赖，缺失时仍允许框架运行
    import mediapipe as mp

    _MP_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖缺失降级路径
    mp = None  # type: ignore
    _MP_AVAILABLE = False

try:  # OpenCV 为可选依赖
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖缺失降级路径
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False


@dataclass
class MediaPipeResult:
    """MediaPipe 单帧检测结果聚合对象。

    Attributes:
        faces: 人脸 Face Mesh 关键点列表。
        pose: 人体姿态关键点（可能为空）。
    """

    faces: List[FaceLandmark] = field(default_factory=list)
    pose: Optional[PoseLandmark] = None


class MediaPipeProcessor:
    """MediaPipe 关键点检测处理器。

    统一管理 Face Mesh 与人脸姿态检测，输出项目标准数据结构。

    Attributes:
        max_num_faces: 最大检测人脸数。
        _face_mesh: MediaPipe Face Mesh 引擎。
        _pose: MediaPipe Pose 引擎。
    """

    def __init__(self) -> None:
        """初始化 MediaPipe 处理器及各引擎。"""
        cfg = get_camera()
        mp_cfg = cfg.get("mediapipe", {})
        self.max_num_faces = int(mp_cfg.get("max_num_faces", 10))
        self.min_detection_confidence = float(
            mp_cfg.get("min_detection_confidence", 0.5)
        )
        self.min_tracking_confidence = float(
            mp_cfg.get("min_tracking_confidence", 0.5)
        )
        self.static_image_mode = bool(mp_cfg.get("static_image_mode", False))
        self.refine_landmarks = bool(mp_cfg.get("refine_landmarks", True))

        self._face_mesh = None
        self._pose = None
        if _MP_AVAILABLE:
            self._init_engines()
        else:
            logger.warning("MediaPipe 未安装，视觉关键点检测将返回空结果")

    def _init_engines(self) -> None:
        """实例化 Face Mesh 与 Pose 引擎（依赖 MediaPipe）。"""
        # 使用 MediaPipe 官方公开 API（mp.solutions），不依赖内部包路径
        face_mesh = mp.solutions.face_mesh
        pose = mp.solutions.pose

        self._face_mesh = face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            refine_landmarks=self.refine_landmarks,
            max_num_faces=self.max_num_faces,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._pose = pose.Pose(
            static_image_mode=self.static_image_mode,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        logger.info(
            "MediaPipe 引擎初始化完成，max_faces=%s", self.max_num_faces
        )

    def process(self, frame: np.ndarray) -> MediaPipeResult:
        """处理单帧图像并返回关键点检测结果。

        Args:
            frame: OpenCV BGR 图像矩阵。

        Returns:
            聚合的 MediaPipe 检测结果对象。
        """
        if frame is None or not _MP_AVAILABLE or not _CV2_AVAILABLE:
            return MediaPipeResult()

        # OpenCV 图像为 BGR，MediaPipe 需要 RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces: List[FaceLandmark] = []
        pose_result: Optional[PoseLandmark] = None

        # Face Mesh 检测
        if self._face_mesh is not None:
            mesh = self._face_mesh.process(rgb)
            if mesh.multi_face_landmarks:
                faces = [
                    self._to_face_landmark(lms, i, rgb.shape[1], rgb.shape[0])
                    for i, lms in enumerate(mesh.multi_face_landmarks)
                ]

        # Pose 检测
        if self._pose is not None:
            p = self._pose.process(rgb)
            if p.pose_landmarks:
                pose_result = self._to_pose_landmark(
                    p.pose_landmarks, rgb.shape[1], rgb.shape[0]
                )

        return MediaPipeResult(faces=faces, pose=pose_result)

    def _to_face_landmark(
        self, landmarks, face_id: int, img_w: int, img_h: int
    ) -> FaceLandmark:
        """将 Face Mesh 原始关键点转换为项目数据结构。

        Args:
            landmarks: MediaPipe 人脸关键点对象。
            face_id: 人脸编号。
            img_w: 图像宽度。
            img_h: 图像高度。

        Returns:
            项目内 FaceLandmark 对象。
        """
        pts = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # 归一化坐标转换为像素坐标
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox = (
            int(min_x * img_w),
            int(min_y * img_h),
            int((max_x - min_x) * img_w),
            int((max_y - min_y) * img_h),
        )
        center_x = ((min_x + max_x) / 2.0) * img_w
        center_y = ((min_y + max_y) / 2.0) * img_h

        return FaceLandmark(
            face_id=face_id,
            confidence=1.0,
            bbox=bbox,
            center_x=float(center_x),
            center_y=float(center_y),
            landmarks=pts,
            roi=None,
        )

    def _to_pose_landmark(
        self, landmarks, img_w: int, img_h: int
    ) -> PoseLandmark:
        """将 Pose 原始关键点转换为项目数据结构。

        Args:
            landmarks: MediaPipe 人体关键点对象。
            img_w: 图像宽度。
            img_h: 图像高度。

        Returns:
            项目内 PoseLandmark 对象。
        """
        pts = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
        visibility = [float(lm.visibility) for lm in landmarks.landmark]
        return PoseLandmark(
            landmarks=pts,
            visibility=visibility,
            confidence=sum(visibility) / len(visibility) if visibility else 0.0,
        )

    def close(self) -> None:
        """释放 MediaPipe 引擎资源。"""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        logger.info("MediaPipe 处理器已关闭")
