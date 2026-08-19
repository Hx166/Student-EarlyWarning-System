"""tests/cv/test_multi_face.py —— 多人逐人分析测试。"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from src.core.models import FaceLandmark
from src.cv import CVProcessor
from src.utils.config import clear_cache


def _face(face_id: int) -> FaceLandmark:
    """构造带 468 个关键点的人脸数据。"""
    return FaceLandmark(
        face_id=face_id,
        confidence=1.0,
        landmarks=[np.zeros(3) for _ in range(468)],
    )


class MultiFaceAnalysisTest(unittest.TestCase):
    """多人逐人分析测试。"""

    def setUp(self):
        """初始化处理器。"""
        clear_cache()
        self.processor = CVProcessor()

    def tearDown(self):
        """关闭处理器并清缓存。"""
        self.processor.close()
        clear_cache()

    def test_process_analyzes_all_faces(self):
        """检测到多张人脸时应逐人分析。"""
        faces = [_face(0), _face(1), _face(2)]
        with mock.patch.object(
            self.processor.face_detector, "detect", return_value=faces
        ):
            result = self.processor.process(
                np.zeros((480, 640, 3), dtype=np.uint8)
            )
        self.assertEqual(len(result.face_analyses), 3)
        self.assertEqual([a.face_id for a in result.face_analyses], [0, 1, 2])
        # 主脸结果保留在顶层字段（向后兼容）
        self.assertIs(result.eye, result.face_analyses[0].eye)
        self.assertIs(result.fatigue, result.face_analyses[0].fatigue)

    def test_process_single_face(self):
        """单张人脸也应产出单条分析。"""
        with mock.patch.object(
            self.processor.face_detector, "detect", return_value=[_face(0)]
        ):
            result = self.processor.process(
                np.zeros((480, 640, 3), dtype=np.uint8)
            )
        self.assertEqual(len(result.face_analyses), 1)
        self.assertEqual(result.face_analyses[0].face_id, 0)

    def test_process_no_faces(self):
        """无人脸时 face_analyses 应为空。"""
        with mock.patch.object(
            self.processor.face_detector, "detect", return_value=[]
        ):
            result = self.processor.process(
                np.zeros((480, 640, 3), dtype=np.uint8)
            )
        self.assertEqual(len(result.face_analyses), 0)


if __name__ == "__main__":
    unittest.main()
