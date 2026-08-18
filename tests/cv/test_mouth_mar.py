"""tests/cv/test_mouth_mar.py —— 嘴部状态（MAR）模块测试。"""

from __future__ import annotations

import unittest

import numpy as np

from src.core.models import FaceLandmark
from src.cv.mouth.mouth_mar import MOUTH_LANDMARKS, MouthProcessor


class MouthProcessorTest(unittest.TestCase):
    """嘴部纵横比与打哈欠检测测试。"""

    def setUp(self):
        """初始化处理器。"""
        self.processor = MouthProcessor()

    def test_compute_mar_open_mouth(self):
        """张嘴几何的 MAR 计算正确。"""
        pts = np.array(
            [
                [0.0, 0.0],   # 左嘴角
                [2.0, 1.0],   # 上唇外
                [2.0, 0.5],   # 上唇内
                [4.0, 0.0],   # 右嘴角
                [2.0, -1.5],  # 下唇内
                [2.0, -2.0],  # 下唇外
            ]
        )
        mar = self.processor.compute_mar(pts)
        self.assertAlmostEqual(mar, 0.625, places=6)

    def test_compute_mar_insufficient_points(self):
        """关键点不足时返回 0。"""
        self.assertEqual(self.processor.compute_mar(None), 0.0)
        self.assertEqual(self.processor.compute_mar(np.zeros((3, 2))), 0.0)

    def test_analyze_yawn_true(self):
        """张嘴人脸应判定为打哈欠。"""
        face = self._build_face(open_mouth=True)
        state = self.processor.analyze(face)
        self.assertTrue(state.is_yawn)
        self.assertGreaterEqual(state.mar, self.processor.mar_threshold)

    def test_analyze_yawn_false(self):
        """闭嘴人脸不应判定为打哈欠。"""
        face = self._build_face(open_mouth=False)
        state = self.processor.analyze(face)
        self.assertFalse(state.is_yawn)

    def test_analyze_no_landmarks(self):
        """无关键点返回默认状态。"""
        state = self.processor.analyze(None)
        self.assertEqual(state.mar, 0.0)
        self.assertFalse(state.is_yawn)

    def _build_face(self, open_mouth: bool) -> FaceLandmark:
        """构造张嘴/闭嘴人脸数据。"""
        lms = [np.zeros(3) for _ in range(468)]
        if open_mouth:
            pts = [
                [0.0, 0.0], [2.0, 1.0], [2.0, 0.5],
                [4.0, 0.0], [2.0, -1.5], [2.0, -2.0],
            ]
        else:
            pts = [
                [0.0, 0.0], [2.0, 0.1], [2.0, 0.05],
                [4.0, 0.0], [2.0, -0.05], [2.0, -0.1],
            ]
        for i, idx in enumerate(MOUTH_LANDMARKS):
            lms[idx] = np.array([pts[i][0], pts[i][1], 0.0])
        return FaceLandmark(face_id=0, confidence=1.0, landmarks=lms)


if __name__ == "__main__":
    unittest.main()
