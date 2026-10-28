"""tests/offline/test_video_clock.py —— 视频时间轴注入机制测试。"""

from __future__ import annotations

import time
import unittest

from src.cv.behavior.clock import (
    VideoClock,
    install_video_clock,
    uninstall_video_clock,
)
from src.cv.behavior.fatigue import FatigueAnalyzer
from src.core.models import EyeState, HeadPoseResult


class VideoClockTest(unittest.TestCase):
    """视频时间轴时钟测试。"""

    def tearDown(self) -> None:
        """确保每个测试后卸载视频时钟。"""
        uninstall_video_clock()

    def test_from_fps_produces_video_time(self) -> None:
        """固定帧率时钟返回帧序号 / fps。"""
        clock = VideoClock.from_fps(30.0)
        self.assertAlmostEqual(clock.monotonic(), 0.0, places=6)
        self.assertAlmostEqual(clock.monotonic(), 1.0 / 30.0, places=6)
        self.assertAlmostEqual(clock.monotonic(), 2.0 / 30.0, places=6)

    def test_from_timestamps_uses_given_ts(self) -> None:
        """使用给定帧时间戳序列。"""
        clock = VideoClock.from_timestamps([0.1, 0.2, 9.5])
        self.assertAlmostEqual(clock.monotonic(), 0.1, places=6)
        self.assertAlmostEqual(clock.monotonic(), 0.2, places=6)
        self.assertAlmostEqual(clock.monotonic(), 9.5, places=6)

    def test_reset(self) -> None:
        """reset 后回到起点。"""
        clock = VideoClock.from_fps(10.0)
        clock.monotonic()
        clock.monotonic()
        clock.reset()
        self.assertAlmostEqual(clock.monotonic(), 0.0, places=6)

    def test_install_and_uninstall_restores_monotonic(self) -> None:
        """卸载后 time.monotonic 恢复真实时钟。"""
        clock = VideoClock.from_fps(30.0)
        install_video_clock(clock)
        self.assertAlmostEqual(time.monotonic(), 0.0, places=1)
        uninstall_video_clock()
        self.assertGreater(time.monotonic(), 0.0)

    def test_double_install_raises(self) -> None:
        """重复安装视频时钟应报错。"""
        from src.cv.behavior.clock import _ACTIVE_PATCH
        clock = VideoClock.from_fps(30.0)
        install_video_clock(clock)
        with self.assertRaises(RuntimeError):
            install_video_clock(VideoClock.from_fps(10.0))


class FatigueWithVideoClockTest(unittest.TestCase):
    """FatigueAnalyzer + 视频时间轴的集成测试。"""

    def tearDown(self) -> None:
        """卸载视频时钟。"""
        uninstall_video_clock()

    def test_fatigue_window_uses_video_time_not_wall_clock(self) -> None:
        """疲劳 60 秒窗口按视频时间推进，而不是程序运行时间。"""
        # 模拟 10 秒、30fps 视频：全部低头
        fps = 30.0
        clock = VideoClock.from_fps(fps)
        install_video_clock(clock)
        analyzer = FatigueAnalyzer()
        head = HeadPoseResult(pose_status="LookDown", confidence=1.0)

        results = []
        for _ in range(int(10 * fps)):
            results.append(analyzer.analyze(head_pose=head))

        # 10 秒低头 → 低头时长约 10 秒，疲劳分数应显著高于实时快进时的数值
        head_down = results[-1].head_down_duration
        self.assertGreater(head_down, 9.0)
        self.assertLess(head_down, 10.5)
        self.assertGreater(results[-1].fatigue_score, 0.0)

    def test_blink_rate_scaled_to_video_time(self) -> None:
        """眨眼率按视频时间换算。"""
        clock = VideoClock.from_fps(30.0)
        install_video_clock(clock)
        analyzer = FatigueAnalyzer()
        eye = EyeState(blink=True, perclos=0.0, confidence=1.0)
        for _ in range(60):
            analyzer.analyze(eye_state=eye)
        result = analyzer.analyze(eye_state=eye)
        self.assertGreater(result.blink_rate, 50.0)


if __name__ == "__main__":
    unittest.main()
