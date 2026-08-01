"""src/camera/camera.py —— 摄像头采集实现。

通过 OpenCV VideoCapture 完成视频帧的实时采集，
并统计 FPS、处理自动重连等异常情况。
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np

from src.config import get_camera
from src.core.models import Frame
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CameraError(Exception):
    """摄像头异常。

    Raises:
        当摄像头初始化或读取失败时抛出。
    """


class Camera:
    """摄像头采集器。

    封装 OpenCV VideoCapture，提供帧读取、FPS 统计与自动重连能力。

    Attributes:
        index: 摄像头索引。
        width: 采集宽度。
        height: 采集高度。
        fps: 目标帧率。
    """

    def __init__(self, index: int = 0) -> None:
        """初始化摄像头采集器。

        Args:
            index: 摄像头索引，默认 0。
        """
        cfg = get_camera()
        self.index = int(cfg.get("camera", {}).get("index", index))
        self.width = int(cfg.get("camera", {}).get("width", 1280))
        self.height = int(cfg.get("camera", {}).get("height", 720))
        self.fps = int(cfg.get("camera", {}).get("fps", 30))
        self.flip = bool(cfg.get("camera", {}).get("flip", False))
        self.auto_reconnect = bool(
            cfg.get("camera", {}).get("auto_reconnect", True)
        )
        self.reconnect_interval = float(
            cfg.get("camera", {}).get("reconnect_interval", 2.0)
        )

        self._capture = None
        self._frame_id = 0
        self._last_time: Optional[float] = None
        self._current_fps: float = 0.0
        self._lock = threading.Lock()

    def open(self) -> None:
        """打开摄像头连接。

        Raises:
            CameraError: 打开摄像头失败。
        """
        try:
            import cv2
        except ImportError as exc:  # 缺少 cv2 时给出清晰错误
            raise CameraError("未安装 opencv-python，请先安装依赖") from exc

        self._capture = cv2.VideoCapture(self.index)
        if self._capture is None or not self._capture.isOpened():
            self._capture = None
            raise CameraError(f"无法打开摄像头，索引: {self.index}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info("摄像头已打开: index=%s", self.index)

    def read(self) -> Optional[Frame]:
        """读取一帧视频数据。

        Returns:
            当前帧数据；读取失败且无法重连时返回 None 。
        """
        if self._capture is None:
            return None

        try:
            import cv2
        except ImportError as exc:  # 防御：缺少 cv2
            raise CameraError("未安装 opencv-python，请先安装依赖") from exc

        ret, image = self._capture.read()
        if not ret or image is None:
            logger.warning("读取摄像头帧失败")
            if self.auto_reconnect:
                self._try_reconnect()
            return None

        if self.flip:
            image = cv2.flip(image, 1)

        now = time.time()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                self._current_fps = 1.0 / dt
        self._last_time = now

        with self._lock:
            self._frame_id += 1
            frame = Frame(
                frame_id=self._frame_id,
                timestamp=datetime.now(),
                image=image,
            )

        return frame

    def _try_reconnect(self) -> None:
        """尝试重新打开摄像头。"""
        try:
            self.close()
            time.sleep(self.reconnect_interval)
            self.open()
            logger.info("摄像头重连成功")
        except CameraError as exc:
            logger.error("摄像头重连失败: %s", exc)

    @property
    def current_fps(self) -> float:
        """当前采集帧率。

        Returns:
            最近统计的 FPS 值。
        """
        return self._current_fps

    @property
    def frame_id(self) -> int:
        """当前帧编号。

        Returns:
            最近读取帧的编号。
        """
        with self._lock:
            return self._frame_id

    def is_opened(self) -> bool:
        """判断摄像头是否处于打开状态。

        Returns:
            已打开返回 True，否则 False。
        """
        if self._capture is None:
            return False
        try:
            import cv2
            return bool(self._capture.isOpened())
        except ImportError:
            return False

    def close(self) -> None:
        """释放摄像头资源。"""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception as exc:  # pragma: no cover - 资源释放尽力而为
                logger.warning("释放摄像头时发生异常: %s", exc)
            self._capture = None
        logger.info("摄像头已关闭")

    def __enter__(self) -> "Camera":
        """上下文管理器进入。

        Returns:
            自身实例。
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器退出，释放资源。"""
        self.close()
