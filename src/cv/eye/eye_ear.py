"""src/cv/eye/eye_ear.py —— 眼部状态（EyeEAR）检测模块。

基于 MediaPipe Face Mesh 468 关键点计算眼睛纵横比（EAR），
判断眼睛开闭状态并进行眨眼检测，同时维护 PERCLOS 疲劳统计指标。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from src.core.models import EyeState, FaceLandmark
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 左眼 MediaPipe 关键点编号（EyeEAR 设计文档定义）
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
# 右眼 MediaPipe 关键点编号
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]


class EyeEarProcessor:
    """眼部纵横比（EyeEAR）处理器。

    负责计算左右眼 EAR、判定睁闭眼状态、检测眨眼事件，
    并维护滑动窗口内的 PERCLOS 指标。

    Attributes:
        ear_threshold: 闭眼判定阈值。
        blink_min_frames: 眨眼最小连续闭眼帧数。
        perclos_window: PERCLOS 统计窗口（帧数）。
    """

    def __init__(self, mediapipe=None) -> None:
        """初始化眼部状态处理器。

        Args:
            mediapipe: 共享的 MediaPipe 处理器引用（可选，暂存备用）。
        """
        self.mediapipe = mediapipe
        self.ear_threshold = 0.20
        self.blink_min_frames = 3
        self.perclos_window = 300
        # 状态机：连续闭眼帧计数
        self._closed_frames = 0
        # 滑动窗口内的睁闭眼标记（True=睁眼，False=闭眼）
        self._eye_frame_states: deque = deque(maxlen=self.perclos_window)
        logger.info(
            "EyeEarProcessor 已初始化，阈值=%s", self.ear_threshold
        )

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

    def _landmark_points(self, landmarks, indices) -> Optional[np.ndarray]:
        """提取指定编号的人脸关键点像素/归一化坐标。

        Args:
            landmarks: FaceLandmark 对象的 landmarks 列表。
            indices: 需要提取的关键点编号列表。

        Returns:
            关键点坐标数组；数据缺失或越界时返回 None。
        """
        if landmarks is None:
            return None
        points = []
        for idx in indices:
            if idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            points.append([lm[0], lm[1], lm[2] if len(lm) > 2 else 0.0])
        return np.asarray(points, dtype=np.float64)

    def analyze(
        self, landmarks: Optional[FaceLandmark] = None, frame: np.ndarray = None
    ) -> EyeState:
        """分析眼部状态。

        Args:
            landmarks: 人脸关键点对象（FaceLandmark）。
            frame: 视频帧（保留，供扩展）。

        Returns:
            眼部状态结果对象。
        """
        # 统一输入：仅接受 FaceLandmark 标准结构
        if not isinstance(landmarks, FaceLandmark):
            return self._build_state(0.5, 0.5, 0.5, "unknown", False)

        lms = landmarks.landmarks
        if lms is None:
            # 无人脸时返回保持状态
            return self._build_state(0.5, 0.5, 0.5, "unknown", False)

        left_pts = self._landmark_points(lms, LEFT_EYE_LANDMARKS)
        right_pts = self._landmark_points(lms, RIGHT_EYE_LANDMARKS)

        left_ear = self.compute_ear(left_pts)
        right_ear = self.compute_ear(right_pts)

        # 双眼融合策略：优先使用平均值，单眼缺失时使用另一侧
        if left_pts is not None and right_pts is not None:
            average_ear = (left_ear + right_ear) / 2.0
        elif left_pts is not None:
            average_ear = left_ear
        elif right_pts is not None:
            average_ear = right_ear
        else:
            average_ear = 0.0

        # 判定睁闭眼
        is_open = average_ear >= self.ear_threshold
        eye_state = "open" if is_open else "closed"

        # 更新滑动窗口 PERCLOS 统计
        self._eye_frame_states.append(is_open)
        perclos = self._compute_perclos()

        # 眨眼事件检测（状态机）
        blink = self._update_blink_state(is_open)

        confidence = 1.0 if (left_pts is not None and right_pts is not None) else 0.5
        return self._build_state(
            left_ear, right_ear, average_ear, eye_state, blink,
            perclos=perclos, confidence=confidence,
        )

    def _update_blink_state(self, is_open: bool) -> bool:
        """基于状态机更新眨眼检测。

        Args:
            is_open: 当前帧是否睁眼。

        Returns:
            当前帧是否判定为一次眨眼。
        """
        if is_open:
            # 睁眼时若此前连续闭眼帧数处于眨眼区间，则判定为一次眨眼
            blink_detected = (
                self.blink_min_frames <= self._closed_frames <= 10
            )
            self._closed_frames = 0
            return blink_detected
        # 闭眼帧递增
        self._closed_frames += 1
        return False

    def _compute_perclos(self) -> float:
        """计算窗口内 PERCLOS 指标。

        Returns:
            PERCLOS 值（闭眼帧占比，0~1）。
        """
        if not self._eye_frame_states:
            return 0.0
        closed = sum(1 for s in self._eye_frame_states if not s)
        return closed / len(self._eye_frame_states)

    def _build_state(
        self,
        left_ear: float,
        right_ear: float,
        average_ear: float,
        eye_state: str,
        blink: bool,
        perclos: float = 0.0,
        confidence: float = 0.0,
    ) -> EyeState:
        """构造眼部状态结果对象。

        Args:
            left_ear: 左眼 EAR。
            right_ear: 右眼 EAR。
            average_ear: 平均 EAR。
            eye_state: 睁闭眼状态。
            blink: 是否发生眨眼。
            perclos: PERCLOS 指标。
            confidence: 可信度。

        Returns:
            眼部状态结果对象。
        """
        return EyeState(
            left_ear=left_ear,
            right_ear=right_ear,
            average_ear=average_ear,
            eye_state=eye_state,
            blink=blink,
            perclos=perclos,
            confidence=confidence,
        )
