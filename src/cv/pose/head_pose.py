"""src/cv/pose/head_pose.py —— 头部姿态（Head Pose）估计模块。

基于 MediaPipe Face Mesh 关键点，结合 OpenCV PnP 算法估计
头部三维姿态角（Pitch / Yaw / Roll），并输出基础姿态分类。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.core.models import FaceLandmark, HeadPoseResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:  # OpenCV 为可选依赖
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖缺失降级路径
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

# 头部姿态所需的 Face Mesh 关键点编号（HeadPose 设计文档定义）
HEAD_POSE_LANDMARKS = [1, 33, 263, 61, 291, 152]

# 对应关键点的三维人脸模型坐标（近似标准模型，单位：mm）
_3D_FACE_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],       # 鼻尖
        [-30.0, -30.0, -10.0],  # 左眼
        [30.0, -30.0, -10.0],   # 右眼
        [-40.0, 20.0, -10.0],   # 左嘴角
        [40.0, 20.0, -10.0],    # 右嘴角
        [0.0, 70.0, -30.0],     # 下巴
    ],
    dtype=np.float64,
)

# 姿态分类阈值（参考 HeadPose 设计文档）
_PITCH_DOWN_THRESHOLD = 20.0
_YAW_SIDE_THRESHOLD = 25.0
_ROLL_TILT_THRESHOLD = 20.0

# 默认图像尺寸（MediaPipe 设计文档推荐 640x480，未提供 frame 时使用）
_DEFAULT_IMAGE_WIDTH = 640
_DEFAULT_IMAGE_HEIGHT = 480


class HeadPoseEstimator:
    """头部姿态估计器。

    基于 PnP 算法计算头部俯仰角、偏航角与翻滚角，
    并根据角度阈值输出基础姿态分类。

    Attributes:
        head_down_threshold: 低头判定 Pitch 阈值。
    """

    def __init__(self, mediapipe=None) -> None:
        """初始化头部姿态估计器。

        Args:
            mediapipe: 共享的 MediaPipe 处理器引用（可选，暂存备用）。
        """
        self.mediapipe = mediapipe
        self.head_down_threshold = _PITCH_DOWN_THRESHOLD
        logger.info("HeadPoseEstimator 已初始化")

    def estimate(
        self, landmarks: Optional[FaceLandmark] = None, frame: np.ndarray = None
    ) -> HeadPoseResult:
        """估计头部姿态角并输出姿态分类。

        Args:
            landmarks: 人脸关键点对象（FaceLandmark）。
            frame: 视频帧（可选，用于获取相机内参维度）。

        Returns:
            头部姿态结果对象；输入不合法时返回置信度为 0 的结果。
        """
        if not _CV2_AVAILABLE:
            logger.warning("OpenCV 未安装，无法进行头部姿态估计")
            return HeadPoseResult(confidence=0.0)

        # 统一输入：仅接受 FaceLandmark 标准结构
        if not isinstance(landmarks, FaceLandmark):
            return HeadPoseResult(confidence=0.0)

        lms = landmarks.landmarks
        if lms is None or len(lms) < max(HEAD_POSE_LANDMARKS):
            return HeadPoseResult(confidence=0.0)

        # 图像尺寸：优先使用 frame 维度
        if frame is not None and frame.size > 0:
            image_h, image_w = frame.shape[:2]
        else:
            image_w, image_h = _DEFAULT_IMAGE_WIDTH, _DEFAULT_IMAGE_HEIGHT

        image_points = []
        for idx in HEAD_POSE_LANDMARKS:
            lm = lms[idx]
            image_points.append([lm[0] * image_w, lm[1] * image_h])
        image_points = np.asarray(image_points, dtype=np.float64)

        # 近似相机内参
        camera_matrix = np.array(
            [
                [image_w, 0.0, image_w / 2.0],
                [0.0, image_w, image_h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))

        # PnP 求解旋转向量与平移向量
        try:
            success, rotation_vec, _ = cv2.solvePnP(
                _3D_FACE_MODEL, image_points, camera_matrix, dist_coeffs
            )
        except cv2.error as exc:  # 数值异常时返回默认姿态
            logger.warning("PnP 求解失败: %s", exc)
            return HeadPoseResult(confidence=0.0)

        if not success:
            return HeadPoseResult(confidence=0.0)

        # 旋转向量转旋转矩阵再转欧拉角
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)

        pose_status = self._classify(pitch, yaw, roll)

        logger.debug(
            "HeadPose: pitch=%.2f yaw=%.2f roll=%.2f status=%s",
            pitch, yaw, roll, pose_status,
        )
        return HeadPoseResult(
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
            pose_status=pose_status,
            confidence=1.0,
        )

    def _classify(self, pitch: float, yaw: float, roll: float) -> str:
        """根据欧拉角进行基础姿态分类。

        Args:
            pitch: 俯仰角（度）。
            yaw: 偏航角（度）。
            roll: 翻滚角（度）。

        Returns:
            姿态分类字符串，取值 Normal/LookDown/LookLeft/LookRight/Tilt。
        """
        # 低头优先判定
        if pitch >= self.head_down_threshold:
            return "LookDown"
        # 左右偏头
        if yaw >= _YAW_SIDE_THRESHOLD:
            return "LookLeft"
        if yaw <= -_YAW_SIDE_THRESHOLD:
            return "LookRight"
        # 身体倾斜
        if abs(roll) >= _ROLL_TILT_THRESHOLD:
            return "Tilt"
        return "Normal"

    def _rotation_matrix_to_euler(
        self, rotation_matrix: np.ndarray
    ) -> tuple:
        """将旋转矩阵转换为欧拉角。

        Args:
            rotation_matrix: 3x3 旋转矩阵。

        Returns:
            (pitch, yaw, roll) 欧拉角元组，单位为度。
        """
        sy = float(np.sqrt(
            rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
        ))
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(
                rotation_matrix[2, 1], rotation_matrix[2, 2]
            )
            yaw = np.arctan2(
                -rotation_matrix[2, 0], sy
            )
            roll = np.arctan2(
                rotation_matrix[1, 0], rotation_matrix[0, 0]
            )
        else:
            pitch = np.arctan2(
                -rotation_matrix[1, 2], rotation_matrix[1, 1]
            )
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0.0

        return (
            np.degrees(pitch),
            np.degrees(yaw),
            np.degrees(roll),
        )
