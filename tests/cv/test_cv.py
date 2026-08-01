"""tests/cv/test_cv.py —— 视觉模块测试。"""

from __future__ import annotations

import unittest

from src.core.models import (
    EngagementResult,
    EyeState,
    FatigueResult,
    HeadPoseResult,
)
from src.cv import CVProcessor
from src.utils.config import clear_cache


class CVProcessorTest(unittest.TestCase):
    """CVProcessor 流水线测试。"""

    def setUp(self):
        """清空配置缓存并初始化处理器。"""
        clear_cache()
        self.processor = CVProcessor()

    def tearDown(self):
        """关闭处理器。"""
        self.processor.close()
        clear_cache()

    def test_process_empty_frame(self):
        """空帧处理不崩溃并返回默认结果。"""
        import numpy as np
        result = self.processor.process(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsNotNone(result.engagement)
        self.assertEqual(result.engagement.engagement_score, 80.0)

    def test_default_results_types(self):
        """各结果对象类型正确。"""
        self.assertIsInstance(self.processor.face_detector, object)
        result = self.processor.process(None)
        self.assertIsInstance(result.eye, EyeState)
        self.assertIsInstance(result.head_pose, HeadPoseResult)
        self.assertIsInstance(result.fatigue, FatigueResult)
        self.assertIsInstance(result.engagement, EngagementResult)


if __name__ == "__main__":
    unittest.main()
