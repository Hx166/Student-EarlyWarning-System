"""src/cv/behavior/offline_fatigue.py —— 离线疲劳分析适配器。

复用生产 `FatigueAnalyzer` 的完整加权融合算法，仅在时间基准上
注入视频时间轴（`VideoClock`），解决离线逐帧快进时
``time.monotonic()`` 与视频时间不一致导致的 60 秒窗口失真问题。

设计约束：

- 不复制一套疲劳算法，直接复用 `FatigueAnalyzer`；
- 时间轴只在 analyze 调用期间注入，避免污染其他模块；
- 每个视频片段应使用独立的离线适配器实例，防止跨片段状态串扰。
"""

from __future__ import annotations

from typing import Optional

from src.core.models import EyeState, FatigueResult, HeadPoseResult, MouthState
from src.cv.behavior.clock import VideoClock, install_video_clock, uninstall_video_clock
from src.cv.behavior.fatigue import FatigueAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OfflineFatigueAnalyzer:
    """按视频时间轴工作的疲劳分析适配器。

    Attributes:
        analyzer: 生产疲劳分析器实例（复用其完整算法）。
        clock: 视频时间轴时钟。
    """

    def __init__(
        self,
        analyzer: Optional[FatigueAnalyzer] = None,
        clock: Optional[VideoClock] = None,
        fps: float = 30.0,
    ) -> None:
        """初始化离线疲劳分析适配器。

        Args:
            analyzer: 生产疲劳分析器；为空时自行创建。
            clock: 视频时间轴时钟；为空时按 fps 创建。
            fps: 视频帧率（clock 为空时使用）。
        """
        self.analyzer = analyzer if analyzer is not None else FatigueAnalyzer()
        self.clock = clock if clock is not None else VideoClock.from_fps(fps)

    def analyze(
        self,
        eye_state: Optional[EyeState] = None,
        head_pose: Optional[HeadPoseResult] = None,
        mouth_state: Optional[MouthState] = None,
    ) -> FatigueResult:
        """按视频时间评估疲劳状态。

        Args:
            eye_state: 眼部状态结果（可为 None）。
            head_pose: 头部姿态结果（可为 None）。
            mouth_state: 嘴部状态结果（可为 None）。

        Returns:
            疲劳评估结果对象。
        """
        install_video_clock(self.clock)
        try:
            return self.analyzer.analyze(
                eye_state=eye_state,
                head_pose=head_pose,
                mouth_state=mouth_state,
            )
        finally:
            uninstall_video_clock()

    def reset(self) -> None:
        """重置疲劳分析器状态与视频时间轴（用于新视频片段）。"""
        uninstall_video_clock()
        self.analyzer = FatigueAnalyzer()
        self.clock.reset()


__all__ = ["OfflineFatigueAnalyzer"]
