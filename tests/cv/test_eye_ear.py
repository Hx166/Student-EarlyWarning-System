"""tests/cv/test_eye_ear.py —— 眼部状态（EyeEAR）模块测试。"""

from __future__ import annotations

import unittest

import numpy as np

from src.cv.eye.eye_ear import EyeEarProcessor, LEFT_EYE_LANDMARKS
from src.core.models import FaceLandmark


class EyeEarProcessorTest(unittest.TestCase):
    """EyeEAR 计算与状态判定测试。"""

    def setUp(self):
        """初始化处理器。"""
        self.processor = EyeEarProcessor()

    def test_compute_ear_known_geometry(self):
        """已知几何形状的 EAR 计算正确。"""
        pts = np.array(
            [
                [0.0, 0.0],   # p1
                [0.5, 1.0],   # p2
                [3.5, 1.0],   # p3
                [4.0, 0.0],   # p4
                [3.5, -1.0],  # p5
                [0.5, -1.0],  # p6
            ]
        )
        ear = self.processor.compute_ear(pts)
        self.assertAlmostEqual(ear, 0.5, places=6)

    def test_compute_ear_insufficient_points(self):
        """关键点不足时返回 0。"""
        self.assertEqual(self.processor.compute_ear(None), 0.0)
        self.assertEqual(self.processor.compute_ear(np.zeros((3, 2))), 0.0)

    def test_analyze_no_landmarks(self):
        """无关键点时返回 unknown 状态。"""
        state = self.processor.analyze(landmarks=None)
        self.assertEqual(state.eye_state, "unknown")

    def test_analyze_open_eye(self):
        """睁眼状态下 EAR 高于阈值。"""
        # 构造一张 468 点人脸，其中左眼为睁眼几何
        lms = [np.zeros(3) for _ in range(468)]
        # 用睁眼几何填充左眼编号
        eye_pts = np.array(
            [
                [0.0, 0.0], [0.5, 1.0], [3.5, 1.0],
                [4.0, 0.0], [3.5, -1.0], [0.5, -1.0],
            ]
        )
        for i, idx in enumerate(LEFT_EYE_LANDMARKS):
            lms[idx] = np.array([eye_pts[i, 0], eye_pts[i, 1], 0.0])

        face = FaceLandmark(face_id=0, confidence=1.0, landmarks=lms)
        state = self.processor.analyze(landmarks=face)
        self.assertEqual(state.eye_state, "open")
        self.assertGreaterEqual(state.average_ear, self.processor.ear_threshold)

    def test_blink_detection(self):
        """连续闭眼后睁眼判定为一次眨眼。"""
        # 构造闭合眼睛（垂直距离趋近 0，EAR ~ 0）
        lms = [np.zeros(3) for _ in range(468)]
        closed_pts = np.array(
            [
                [0.0, 0.0], [0.5, 0.05], [3.5, 0.05],
                [4.0, 0.0], [3.5, -0.05], [0.5, -0.05],
            ]
        )
        for i, idx in enumerate(LEFT_EYE_LANDMARKS):
            lms[idx] = np.array([closed_pts[i, 0], closed_pts[i, 1], 0.0])
        closed_face = FaceLandmark(face_id=0, confidence=1.0, landmarks=lms)

        blink = False
        # 多帧闭眼
        for _ in range(5):
            state = self.processor.analyze(landmarks=closed_face)
            self.assertFalse(state.blink)  # 闭眼中不判定完成
        # 恢复到睁眼，触发眨眼
        open_face = self._build_face(closed=False)
        state = self.processor.analyze(landmarks=open_face)
        self.assertTrue(state.blink)

    def test_perclos_window(self):
        """PERCLOS 反映窗口内闭眼比例。"""
        closed_face = self._build_face(closed=True)
        repeated = 0
        for _ in range(10):
            state = self.processor.analyze(landmarks=closed_face)
            if state.eye_state == "closed":
                repeated += 1
        self.assertAlmostEqual(state.perclos, 1.0, places=6)

    def _build_face(self, closed: bool) -> FaceLandmark:
        """构造睁/闭眼人脸数据。"""
        lms = [np.zeros(3) for _ in range(468)]
        if closed:
            pts = np.array(
                [
                    [0.0, 0.0], [0.5, 0.05], [3.5, 0.05],
                    [4.0, 0.0], [3.5, -0.05], [0.5, -0.05],
                ]
            )
        else:
            pts = np.array(
                [
                    [0.0, 0.0], [0.5, 1.0], [3.5, 1.0],
                    [4.0, 0.0], [3.5, -1.0], [0.5, -1.0],
                ]
            )
        for i, idx in enumerate(LEFT_EYE_LANDMARKS):
            lms[idx] = np.array([pts[i, 0], pts[i, 1], 0.0])
        return FaceLandmark(face_id=0, confidence=1.0, landmarks=lms)


if __name__ == "__main__":
    unittest.main()
