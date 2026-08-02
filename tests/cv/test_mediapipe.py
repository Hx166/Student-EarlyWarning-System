"""tests/cv/test_mediapipe.py —— MediaPipe 关键点模块测试。"""

from __future__ import annotations

import unittest

import numpy as np

from src.cv.mediapipe import MediaPipeProcessor
from src.utils.config import clear_cache


class MediaPipeProcessorTest(unittest.TestCase):
    """MediaPipe 处理器测试。"""

    def setUp(self):
        """初始化处理器。"""
        clear_cache()
        self.processor = MediaPipeProcessor()

    def tearDown(self):
        """释放资源。"""
        self.processor.close()
        clear_cache()

    def test_empty_frame_returns_empty(self):
        """空帧或纯黑帧返回空结果且不崩溃。"""
        result = self.processor.process(None)
        self.assertEqual(len(result.faces), 0)

        black = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.processor.process(black)
        self.assertEqual(len(result.faces), 0)

    def test_close_releases_engines(self):
        """关闭后引擎释放。"""
        self.processor.close()
        self.assertIsNone(self.processor._face_mesh)


if __name__ == "__main__":
    unittest.main()
