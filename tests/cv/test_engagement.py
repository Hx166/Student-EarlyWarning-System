"""tests/cv/test_engagement.py —— 课堂参与度（Engagement）模块测试。"""

from __future__ import annotations

import unittest

from src.core.models import EngagementResult, EyeState, FatigueResult, HeadPoseResult
from src.cv.emotion.engagement import EngagementAnalyzer


class EngagementAnalyzerTest(unittest.TestCase):
    """参与度评估器单元测试。"""

    def setUp(self):
        """初始化评估器。"""
        self.analyzer = EngagementAnalyzer()

    def test_analyze_no_inputs(self):
        """无输入时返回中性分值与等级。"""
        result = self.analyzer.analyze()
        self.assertIsInstance(result, EngagementResult)
        self.assertGreaterEqual(result.engagement_score, 0.0)
        self.assertLessEqual(result.engagement_score, 100.0)
        self.assertIn(result.engagement_level, ("high", "normal", "distracted", "low", "absent"))

    def test_high_engagement(self):
        """高注意输入产生高参与分。"""
        eye = EyeState(perclos=0.0, confidence=1.0)
        head = HeadPoseResult(pose_status="Normal", confidence=1.0)
        fatigue = FatigueResult(fatigue_score=0.0, confidence=1.0)
        result = self.analyzer.analyze(eye_state=eye, head_pose=head, fatigue=fatigue)
        self.assertGreaterEqual(result.engagement_score, 80.0)
        self.assertEqual(result.engagement_level, "high")

    def test_low_engagement(self):
        """明显疲劳与零人脸降低参与分。"""
        # 高 perclos 且没有可靠头部/眼部在场
        eye = EyeState(perclos=1.0, confidence=0.0)
        head = HeadPoseResult(pose_status="LookDown", confidence=0.0)
        fatigue = FatigueResult(fatigue_score=100.0, confidence=1.0)
        result = self.analyzer.analyze(eye_state=eye, head_pose=head, fatigue=fatigue)
        self.assertLess(result.engagement_score, 60.0)

    def test_eye_score(self):
        """眼部组件评分随 PERCLOS 反向变化。"""
        self.assertAlmostEqual(EngagementAnalyzer._eye_score(EyeState(perclos=0.0)), 100.0, places=2)
        self.assertAlmostEqual(EngagementAnalyzer._eye_score(EyeState(perclos=0.5)), 50.0, places=2)

    def test_head_pose_score(self):
        """头部姿态组件评分。"""
        self.assertEqual(EngagementAnalyzer._head_pose_score(HeadPoseResult(pose_status="Normal", confidence=1.0)), 100.0)
        self.assertEqual(EngagementAnalyzer._head_pose_score(HeadPoseResult(pose_status="LookDown", confidence=1.0)), 45.0)

    def test_fatigue_component(self):
        """疲劳组件评分与 fatigue_score 反向。"""
        self.assertAlmostEqual(EngagementAnalyzer._fatigue_component(FatigueResult(fatigue_score=30.0)), 70.0, places=2)

    def test_level_boundaries(self):
        """等级划分边界正确。"""
        self.assertEqual(EngagementAnalyzer._classify_level(90), "high")
        self.assertEqual(EngagementAnalyzer._classify_level(70), "normal")
        self.assertEqual(EngagementAnalyzer._classify_level(50), "distracted")
        self.assertEqual(EngagementAnalyzer._classify_level(30), "low")
        self.assertEqual(EngagementAnalyzer._classify_level(10), "absent")

    def test_confidence_average(self):
        """可信度为各输入可信度平均值。"""
        conf = EngagementAnalyzer._compute_confidence(
            EyeState(confidence=0.8), HeadPoseResult(confidence=0.6), FatigueResult(confidence=0.4)
        )
        self.assertAlmostEqual(conf, 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
