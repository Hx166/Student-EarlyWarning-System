"""tests/cv/test_head_pose.py —— 头部姿态（HeadPose）模块测试。"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

from src.cv.pose.head_pose import (
    HeadPoseEstimator,
    HEAD_POSE_LANDMARKS,
    _3D_FACE_MODEL,
)
from src.core.models import FaceLandmark


@unittest.skipUnless(_CV2, "需要 OpenCV")
class HeadPoseEstimatorTest(unittest.TestCase):
    """HeadPose 估计测试。"""

    def setUp(self):
        """初始化估计器。"""
        self.estimator = HeadPoseEstimator()
        self.img_w = 640
        self.img_h = 480

    def test_no_landmarks_returns_zero_confidence(self):
        """无关键点时返回置信度为 0 的结果。"""
        result = self.estimator.estimate(landmarks=None)
        self.assertEqual(result.confidence, 0.0)

    def test_frontal_pose_returns_near_zero_angles(self):
        """正视角投影的人脸关键点应得到接近 0 的姿态角。"""
        # 通过相机投影标准 3D 模型生成正视角的 2D 像素点
        camera_matrix = np.array(
            [
                [self.img_w, 0.0, self.img_w / 2.0],
                [0.0, self.img_w, self.img_h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1))
        rvec = np.zeros((3, 1))  # 无旋转 = 正视角
        tvec = np.array([[0.0], [0.0], [50.0]])  # 置于相机前
        projected, _ = cv2.projectPoints(
            _3D_FACE_MODEL, rvec, tvec, camera_matrix, dist
        )
        projected = projected.reshape(-1, 2)

        # 归一化到 [0,1] 以符合 FaceLandmark 存储约定
        lms = [np.zeros(3) for _ in range(468)]
        for i, idx in enumerate(HEAD_POSE_LANDMARKS):
            x = float(projected[i, 0]) / self.img_w
            y = float(projected[i, 1]) / self.img_h
            lms[idx] = np.array([x, y, 0.0])

        face = FaceLandmark(face_id=0, confidence=1.0, landmarks=lms)
        result = self.estimator.estimate(
            landmarks=face, frame=np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        )
        self.assertEqual(result.confidence, 1.0)
        self.assertAlmostEqual(result.pitch, 0.0, delta=5.0)
        self.assertAlmostEqual(result.yaw, 0.0, delta=5.0)
        self.assertAlmostEqual(result.roll, 0.0, delta=5.0)

    def test_missing_landmarks_returns_low_confidence(self):
        """关键点数量不足时返回低置信度。"""
        lms = [np.zeros(3) for _ in range(100)]  # 不足 291
        face = FaceLandmark(face_id=0, confidence=1.0, landmarks=lms)
        result = self.estimator.estimate(
            landmarks=face, frame=np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        )
        self.assertEqual(result.confidence, 0.0)



class PoseClassificationTest(unittest.TestCase):
    """姿态分类逻辑测试（不依赖 OpenCV）。"""

    def setUp(self):
        """初始化估计器。"""
        self.estimator = HeadPoseEstimator()

    def test_normal(self):
        """小角度返回 Normal。"""
        self.assertEqual(self.estimator._classify(0.0, 0.0, 0.0), "Normal")

    def test_look_down(self):
        """低头返回 LookDown。"""
        self.assertEqual(self.estimator._classify(35.0, 5.0, 5.0), "LookDown")

    def test_look_left(self):
        """向右偏头返回 LookLeft。"""
        self.assertEqual(self.estimator._classify(5.0, 30.0, 5.0), "LookLeft")

    def test_look_right(self):
        """向左偏头返回 LookRight。"""
        self.assertEqual(self.estimator._classify(5.0, -30.0, 5.0), "LookRight")

    def test_tilt(self):
        """身体倾斜返回 Tilt。"""
        self.assertEqual(self.estimator._classify(5.0, 5.0, 25.0), "Tilt")

if __name__ == "__main__":
    unittest.main()
