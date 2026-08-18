"""src/cv/face/detector.py —— 人脸检测模块。

负责在视频帧中定位学生人脸并输出人脸关键点数据结构。
基于 MediaPipe Face Mesh 实现，复用共享的 MediaPipeProcessor 实例，
并补充人脸 ROI 区域的提取。

MediaPipe 处理器由外部（CVProcessor）创建并通过依赖注入传入，
避免在多个模块中重复初始化模型引擎。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.config import get_camera
from src.core.models import FaceLandmark, PoseLandmark
from src.cv.mediapipe import MediaPipeProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """人脸检测器。

    基于 MediaPipe Face Mesh 定位人脸，并输出统一的关键点数据结构。
    复用外部注入的 MediaPipeProcessor，避免重复初始化模型引擎。

    Attributes:
        mediapipe: MediaPipe 关键点检测处理器。
        max_faces: 最大检测人脸数。
        min_confidence: 最小检测置信度。
        _owns_mediapipe: 是否由本类负责关闭 MediaPipe 实例。
    """

    def __init__(self, mediapipe: Optional[MediaPipeProcessor] = None) -> None:
        """初始化人脸检测器。

        Args:
            mediapipe: 外部注入的 MediaPipe 处理器；None 时自行创建。
        """
        cfg = get_camera()
        self.max_faces = int(cfg.get("face_detection", {}).get("max_faces", 10))
        self.min_confidence = float(
            cfg.get("face_detection", {}).get("min_detection_confidence", 0.5)
        )
        # 依赖注入：优先复用外部共享实例
        if mediapipe is not None:
            self.mediapipe = mediapipe
            self._owns_mediapipe = False
        else:
            self.mediapipe = MediaPipeProcessor()
            self._owns_mediapipe = True
        self._last_frame: np.ndarray = None
        self._last_pose: Optional[PoseLandmark] = None
        logger.info("FaceDetector 已初始化，max_faces=%s", self.max_faces)

    def detect(self, frame: np.ndarray) -> List[FaceLandmark]:
        """检测图像中的人脸。

        Args:
            frame: OpenCV BGR 图像矩阵。

        Returns:
            人脸关键点结果列表；未检测到人脸时返回空列表。
        """
        if frame is None:
            self._last_frame = None
            self._last_pose = None
            return []

        self._last_frame = frame
        result = self.mediapipe.process(frame)

        # 记录最近一次的人体姿态关键点（供上层消费）
        self._last_pose = result.pose

        # 按置信度过滤并限制人数
        faces = [f for f in result.faces if f.confidence >= self.min_confidence]
        faces = faces[: self.max_faces]

        # 补充人脸 ROI 区域
        for face in faces:
            face.roi = self._extract_roi(frame, face.bbox)

        logger.debug("检测到 %s 张人脸", len(faces))
        return faces

    def _extract_roi(self, frame: np.ndarray, bbox) -> np.ndarray:
        """从原图中提取人脸 ROI 区域。

        Args:
            frame: OpenCV BGR 图像矩阵。
            bbox: 人脸边界框 (x, y, w, h)。

        Returns:
            人脸 ROI 图像矩阵。
        """
        if bbox is None:
            return None
        x, y, w, h = bbox
        # 边界裁剪，防止越界
        height, width = frame.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + w)
        y1 = min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        return frame[y0:y1, x0:x1].copy()

    def get_mediapipe(self) -> MediaPipeProcessor:
        """获取当前复用的 MediaPipe 实例。

        Returns:
            MediaPipe 处理器实例，供其他模块共享引用。
        """
        return self.mediapipe

    def get_last_pose(self) -> Optional[PoseLandmark]:
        """获取最近一次检测的人体姿态关键点。

        Returns:
            姿态关键点对象；无姿态时返回 None。
        """
        return self._last_pose

    def close(self) -> None:
        """释放人脸检测资源。

        仅当本类自建 MediaPipe 实例时才关闭它，避免关闭共享实例。
        """
        if self._owns_mediapipe:
            self.mediapipe.close()
        logger.info("FaceDetector 已关闭")
