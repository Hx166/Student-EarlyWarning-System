"""src/cv/face/detector.py —— 人脸检测模块。

负责在视频帧中定位学生人脸并输出人脸关键点数据。
第一阶段优先保证接口完整，内部算法在第二阶段完善。
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.config import get_camera
from src.core.models import FaceLandmark
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """人脸检测器。

    基于 MediaPipe Face Detection 设计接口。
    第一阶段提供默认返回值保证流程可运行。

    Attributes:
        max_faces: 最大检测人脸数。
    """

    def __init__(self) -> None:
        """初始化人脸检测器。"""
        cfg = get_camera()
        self.max_faces = int(cfg.get("face_detection", {}).get("max_faces", 10))
        self.min_confidence = float(
            cfg.get("face_detection", {}).get("min_detection_confidence", 0.5)
        )
        logger.info("FaceDetector 已初始化，max_faces=%s", self.max_faces)

    def detect(self, frame: np.ndarray) -> List[FaceLandmark]:
        """检测图像中的人脸。

        Args:
            frame: OpenCV BGR 图像矩阵。

        Returns:
            人脸关键点结果列表；未检测到人脸时返回空列表。
        """
        if frame is None:
            return []

        # TODO(阶段二): 接入 MediaPipe Face Detection 完成真实检测
        # 当前返回默认空结果，保证框架可运行
        return []

    def close(self) -> None:
        """释放人脸检测资源。"""
        logger.info("FaceDetector 已关闭")
