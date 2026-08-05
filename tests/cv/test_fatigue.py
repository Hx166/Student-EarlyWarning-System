"""tests/cv/test_fatigue.py —— 疲劳状态（Fatigue）模块测试。"""

from __future__ import annotations

import unittest
from unittest import mock

from src.core.models import EyeState, FatigueResult, HeadPoseResult
from src.cv.behavior.fatigue import FatigueAnalyzer


class FatigueAnalyzerTest(unittest.TestCase):
    """疲劳评估器单元测试。"""

    def setUp(self):
        """初始化分析器与时钟 mock。"""
        self.analyzer = FatigueAnalyzer()
        self.clock = mock.patch("src.cv.behavior.fatigue.time.monotonic")
        self.mock_time = self.clock.start()
        self.mock_time.side_effect = [0.0]

    def tearDown(self):
        """释放 mock。"""
        self.clock.stop()

    def _next(self, ts: float) -> None:
        """设置下一次 time.monotonic 返回值。"""
        self.mock_time.side_effect = list(self.mock_time.side_effect) + [ts]

    def test_analyze_no_inputs_returns_zero(self):
        """无输入时返回零评分与 normal。"""
        result = self.analyzer.analyze()
        self.assertIsInstance(result, FatigueResult)
        self.assertEqual(result.fatigue_score, 0.0)
        self.assertEqual(result.fatigue_level, "normal")
        self.assertEqual(result.perclos, 0.0)

    def test_analyze_high_perclos_raises_score(self):
        """高 PERCLOS 提升疲劳评分。"""
        self._next(0.01)
        eye = EyeState(perclos=0.6, blink=False, confidence=1.0)
        result = self.analyzer._compute_score(
            perclos=eye.perclos, blink_rate=0.0, head_down_duration=0.0
        )
        self.assertAlmostEqual(result, 30.0, places=2)

    def test_analyze_blink_rate_accumulates(self):
        """窗口内眨眼次数转换为每分钟率。"""
        # 第一帧初始化，后续多帧产生眨眼
        self.analyzer._last_ts = None
        self.mock_time.side_effect = None
        self.mock_time.return_value = 0.0
        eye = EyeState(blink=True, perclos=0.0, confidence=1.0)
        for _ in range(10):
            self.analyzer.analyze(eye_state=eye)
        result = self.analyzer.analyze(eye_state=eye)
        # 约10次眨眼在窗口内应大于 0
        self.assertGreater(result.blink_rate, 0.0)

    def test_head_down_accumulates_duration(self):
        """低头状态积累持续时长。"""
        # 每帧间隔 1 秒，持续 5 帧低头
        self.analyzer._last_ts = None
        hook = {"t": 0.0}
        def monotonic():
            hook["t"] += 1.0
            return hook["t"]
        self.mock_time.side_effect = monotonic
        head_pose = HeadPoseResult(
            pose_status="LookDown", confidence=1.0
        )
        for _ in range(5):
            self.analyzer.analyze(head_pose=head_pose)
        result = self.analyzer.analyze(head_pose=head_pose)
        # 累计时长应大于 0
        self.assertGreater(result.head_down_duration, 0.0)
        self.assertGreater(result.fatigue_score, 0.0)

    def test_level_classification(self):
        """等级划分边界正确。"""
        self.assertEqual(FatigueAnalyzer._classify_level(10), "normal")
        self.assertEqual(FatigueAnalyzer._classify_level(40), "mild")
        self.assertEqual(FatigueAnalyzer._classify_level(70), "moderate")
        self.assertEqual(FatigueAnalyzer._classify_level(90), "severe")

    def test_confidence_average(self):
        """可信度为各输入可信度平均值。"""
        eye = EyeState(confidence=0.8)
        head = HeadPoseResult(confidence=0.6)
        conf = FatigueAnalyzer._compute_confidence(eye, head)
        self.assertAlmostEqual(conf, 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
