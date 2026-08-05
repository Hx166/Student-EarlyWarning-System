"""scripts/face_demo.py —— 实时人脸关键点可视化 Demo。

在摄像头画面上实时绘制 468 个 Face Mesh 关键点、人脸框，
并在左上角显示 FPS、EAR、PERCLOS、Pitch/Yaw/Roll 与检测人数。
按 q 或 Esc 退出程序。

Run:
    python scripts/face_demo.py
"""

# Requirements: opencv-python, mediapipe>=0.10,<0.11, numpy

from __future__ import annotations

import os
import sys
import time
from typing import List

import numpy as np

# 将项目根目录加入 sys.path，便于直接运行本脚本
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2

from src.camera import Camera
from src.cv import CVProcessor
from src.core.models import FaceLandmark
from src.utils.config import clear_cache
from src.utils.logger import get_logger, init_logging

logger = get_logger(__name__)

# MediaPipe 官方公开 API：Face Mesh 连接关系（用于绘制关键点连线）
try:
    import mediapipe as mp

    _FACEMESH_CONTOURS = mp.solutions.face_mesh.FACEMESH_CONTOURS
except ImportError:  # pragma: no cover - 缺失媒体依赖
    mp = None  # type: ignore
    _FACEMESH_CONTOURS = set()

# 显示参数
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_SCALE = 0.55
_TEXT_THICKNESS = 1
_WINDOW_NAME = "Student EarlyWarning - Face Mesh Demo"
_DRWA_TITLE = "左上角信息面板: 绘制在画面左上区域"


def _setup_path() -> None:
    """为 Demo 初始化日志系统。

    Returns:
        无返回值。
    """
    init_logging()
    clear_cache()


def _draw_landmarks(frame: np.ndarray, landmarks: List[tuple]) -> None:
    """在画面上绘制 468 个 Face Mesh 关键点与连线。

    Args:
        frame: 待绘制的 OpenCV BGR 图像（原地修改）。
        landmarks: 归一化关键点坐标列表 [(x, y, z), ...]。
    """
    if landmarks is None:
        return

    height, width = frame.shape[:2]
    # 归一化坐标转像素坐标
    points = [
        (int(lm[0] * width), int(lm[1] * height)) for lm in landmarks
    ]

    # 绘制关键点连线（网格）
    for conn in _FACEMESH_CONTOURS:
        p1 = points[conn[0]]
        p2 = points[conn[1]]
        cv2.line(frame, p1, p2, (0, 255, 0), 1)

    # 绘制关键点
    for (px, py) in points:
        cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)


def _draw_bbox(frame: np.ndarray, face: FaceLandmark) -> None:
    """绘制人脸边界框。

    Args:
        frame: 待绘制的 OpenCV BGR 图像（原地修改）。
        face: 人脸关键点结果对象。
    """
    if face.bbox is None:
        return
    x, y, w, h = face.bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def _draw_info_panel(
    frame: np.ndarray,
    fps: float,
    ear: float,
    perclos: float,
    pitch: float,
    yaw: float,
    roll: float,
    face_count: int,
) -> None:
    """在画面左上角绘制实时信息面板。

    Args:
        frame: 待绘制的 OpenCV BGR 图像（原地修改）。
        fps: 当前帧率。
        ear: 平均 EAR。
        perclos: PERCLOS 指标。
        pitch: 俯仰角。
        yaw: 偏航角。
        roll: 翻滚角。
        face_count: 检测到的人数。
    """
    lines = [
        f"FPS    : {fps:.1f}",
        f"EAR    : {ear:.3f}",
        f"PERCLOS: {perclos:.2f}",
        f"Pitch  : {pitch:6.1f} deg",
        f"Yaw    : {yaw:6.1f} deg",
        f"Roll   : {roll:6.1f} deg",
        f"Faces  : {face_count}",
    ]
    # 半透明信息面板背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, 10 + len(lines) * 24), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(lines):
        y_pos = 30 + i * 24
        cv2.putText(
            frame, line, (16, y_pos), _FONT,
            _TEXT_SCALE, (0, 255, 255), _TEXT_THICKNESS, cv2.LINE_AA,
        )


class FaceDemoRunner:
    """实时人脸关键点 Demo 运行器。

    负责摄像头采集、视觉处理与可视化渲染的主循环。

    Attributes:
        camera: 摄像头采集器。
        processor: 视觉处理流水线。
        _fps: 当前帧率估算。
        _last_time: 上一次统计时间。
    """

    def __init__(self, camera_index: int = 0) -> None:
        """初始化 Demo 运行器。

        Args:
            camera_index: 摄像头索引。
        """
        clear_cache()
        self.camera = Camera(index=camera_index)
        self.processor = CVProcessor()
        self._fps = 0.0
        self._last_time = None

    def _measure_fps(self) -> float:
        """计算实时帧率。

        Returns:
            当前估算的 FPS 值。
        """
        now = time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                self._fps = 1.0 / dt
        self._last_time = now
        return self._fps

    def _should_stop(self, key: int) -> bool:
        """判断是否退出程序。

        Args:
            key: OpenCV 等待按键返回值。

        Returns:
            按 q 或 Esc 时返回 True，否则 False。
        """
        return key in (ord("q"), ord("Q"), 27)

    def run(self) -> None:
        """运行实时可视化主循环。

        Returns:
            无返回值。
        """
        self.camera.open()
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
        logger.info("实时 Demo 已启动，按 q 或 Esc 退出")

        try:
            while True:
                frame_obj = self.camera.read()
                if frame_obj is None:
                    continue

                frame = frame_obj.image
                result = self.processor.process(frame)

                # 绘制每张人脸的关键点与边框
                for face in result.faces:
                    _draw_landmarks(frame, face.landmarks)
                    _draw_bbox(frame, face)

                fps = self._measure_fps()
                face_count = len(result.faces)
                _draw_info_panel(
                    frame,
                    fps=fps,
                    ear=result.eye.average_ear,
                    perclos=result.eye.perclos,
                    pitch=result.head_pose.pitch,
                    yaw=result.head_pose.yaw,
                    roll=result.head_pose.roll,
                    face_count=face_count,
                )

                cv2.imshow(_WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if self._should_stop(key):
                    logger.info("用户请求退出")
                    break
        finally:
            cv2.destroyAllWindows()
            self.close()

    def close(self) -> None:
        """释放摄像头与视觉处理资源。

        Returns:
            无返回值。
        """
        self.processor.close()
        self.camera.close()
        logger.info("Demo 已关闭")


if __name__ == "__main__":
    _setup_path()
    demo = FaceDemoRunner()
    demo.run()
