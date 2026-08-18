"""tests/cv/test_fatigue_yawn.py —— 疲劳模块打哈欠分量测试。"""

from __future__ import annotations

import unittest
from unittest import mock

from src.core.models import MouthState
from src.cv.behavior.fatigue import FatigueAnalyzer


class FatigueYawnTest(unittest.TestCase):
    """疲劳评估器打哈欠分量测试。"""

    def setUp(self):
        """初始化分析器。"""
        self.analyzer = FatigueAnalyzer()

    def test_weights_are_normalized(self):
        """四项权重之和应为 1。"""
        total = (
            self.analyzer.perclos_weight
            + self.analyzer.blink_weight
            + self.analyzer.head_down_weight
            + self.analyzer.yawn_weight
        )
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertAlmostEqual(self.analyzer.yawn_weight, 0.15, places=6)

    def test_yawn_full_window_scores_15(self):
        """满窗口打哈欠贡献 15 分。"""
        score = self.analyzer._compute_score(
            perclos=0.0, blink_rate=0.0, head_down_duration=0.0,
            yawn_duration=self.analyzer.fatigue_window,
        )
        self.assertAlmostEqual(score, 15.0, places=2)

    def test_yawn_duration_and_rate_accumulate(self):
        """持续打哈欠应累积时长与频率。"""
        hook = {"t": 0.0}

        def monotonic():
            hook["t"] += 1.0
            return hook["t"]

        with mock.patch(
            "src.cv.behavior.fatigue.time.monotonic", side_effect=monotonic
        ):
            self.analyzer._last_ts = None
            mouth = MouthState(is_yawn=True, mar=0.7, confidence=1.0)
            result = None
            for _ in range(5):
                result = self.analyzer.analyze(mouth_state=mouth)

        self.assertGreater(result.yawn_duration, 0.0)
        self.assertGreater(result.yawn_rate, 0.0)
        self.assertGreater(result.fatigue_score, 0.0)


if __name__ == "__main__":
    unittest.main()
