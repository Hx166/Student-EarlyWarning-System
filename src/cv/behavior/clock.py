"""src/cv/behavior/clock.py —— 疲劳分析时间轴注入机制。

实时推理（FatigueAnalyzer）使用 ``time.monotonic()`` 作为时间基准，
该基准是程序真实运行时间。离线处理视频时若直接复用，60 秒疲劳窗口
将对应“程序处理耗时”而非“视频时间”，导致窗口统计完全失真。

本模块提供：

- ``VideoClock``：统一的视频时间轴抽象（帧序号 → 视频时间戳）；
- ``install_video_clock`` / ``uninstall_video_clock``：以标准库
  ``unittest.mock`` 的方式在 ``FatigueAnalyzer`` 的时间戳来源处注入
  视频时间轴，从而**不修改生产代码**即可离线复现同一套疲劳算法。

约定：

- 视频时间为自视频首帧起的相对秒数（float），单调递增；
- 帧率未知时回退到构造传入的 ``fallback_fps``（默认 30.0）。
"""

from __future__ import annotations

import time
from typing import Optional
from unittest import mock


class VideoClock:
    """视频时间轴时钟。

    Attributes:
        fallback_fps: 未提供帧时间戳时使用的回退帧率。
        _frame_index: 当前帧序号（自 0 开始）。
        _frame_timestamps: 帧时间戳序列（秒，可为空）。
    """

    def __init__(self, fallback_fps: float = 30.0) -> None:
        """初始化视频时间轴时钟。

        Args:
            fallback_fps: 未提供帧时间戳时使用的回退帧率。
        """
        self.fallback_fps = fallback_fps
        self._frame_index = 0
        self._frame_timestamps: list[float] = []

    @classmethod
    def from_fps(cls, fps: float, fallback_fps: Optional[float] = None) -> "VideoClock":
        """按固定帧率创建视频时间轴时钟。

        Args:
            fps: 视频帧率。
            fallback_fps: 回退帧率，缺省与 fps 一致。

        Returns:
            视频时间轴时钟实例。
        """
        resolved = fallback_fps if fallback_fps is not None else fps
        return cls(fallback_fps=resolved)

    @classmethod
    def from_timestamps(
        cls, timestamps: list[float], fallback_fps: float = 30.0
    ) -> "VideoClock":
        """按帧时间戳序列创建视频时间轴时钟。

        Args:
            timestamps: 帧时间戳序列（秒，相对视频起点）。
            fallback_fps: 时间戳不可用时的回退帧率。

        Returns:
            视频时间轴时钟实例。
        """
        inst = cls(fallback_fps=fallback_fps)
        inst._frame_timestamps = list(timestamps)
        return inst

    def monotonic(self) -> float:
        """返回当前帧对应的视频时间（秒）。

        优先使用帧时间戳序列；序列耗尽或格式异常时，
        按回退帧率以帧序号 / fps 推进。

        Returns:
            当前帧的视频时间戳（秒）。
        """
        if self._frame_index < len(self._frame_timestamps):
            ts = self._frame_timestamps[self._frame_index]
            self._frame_index += 1
            if isinstance(ts, (int, float)) and ts >= 0:
                return float(ts)
        ts = self._frame_index / self.fallback_fps
        self._frame_index += 1
        return ts

    def reset(self) -> None:
        """重置帧序号到视频起点。"""
        self._frame_index = 0


_ACTIVE_PATCH: Optional[mock._patch] = None


def install_video_clock(clock: VideoClock) -> None:
    """在 FatigueAnalyzer 的时间戳来源处注入视频时间轴。

    通过标准库 mock 将 ``src.cv.behavior.fatigue.time.monotonic``
    替换为 ``clock.monotonic``；调用前必须先 ``uninstall_video_clock``。

    Args:
        clock: 视频时间轴时钟。

    Raises:
        RuntimeError: 已有视频时钟处于激活状态。
    """
    global _ACTIVE_PATCH
    if _ACTIVE_PATCH is not None:
        raise RuntimeError("视频时间轴时钟已激活，请先 uninstall_video_clock")
    _ACTIVE_PATCH = mock.patch.object(
        time, "monotonic", side_effect=clock.monotonic
    )
    _ACTIVE_PATCH.start()


def uninstall_video_clock() -> None:
    """卸载视频时间轴时钟，恢复 FatigueAnalyzer 默认时间来源。"""
    global _ACTIVE_PATCH
    if _ACTIVE_PATCH is None:
        return
    _ACTIVE_PATCH.stop()
    _ACTIVE_PATCH = None


__all__ = ["VideoClock", "install_video_clock", "uninstall_video_clock"]
