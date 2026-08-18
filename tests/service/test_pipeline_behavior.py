"""tests/service/test_pipeline_behavior.py —— Pipeline 行为事件写入测试。"""

from __future__ import annotations

import unittest
from unittest import mock

from src.core.models import (
    EyeState,
    FaceLandmark,
    HeadPoseResult,
    MouthState,
    PoseLandmark,
)
from src.cv.processor import CVProcessResult
from src.service.pipeline import PipelineService


class _FakeDb:
    """记录 insert_behavior 调用的数据库桩。"""

    def __init__(self):
        self.behaviors = []

    def insert_behavior(self, **kwargs):
        self.behaviors.append(kwargs)
        return len(self.behaviors)


class PipelineBehaviorTest(unittest.TestCase):
    """Pipeline 行为事件写入测试。"""

    def setUp(self):
        """构建带行为跟踪的 pipeline 与桩数据库。"""
        self.camera = mock.MagicMock()
        self.cv = mock.MagicMock()
        self.db = _FakeDb()
        self.pipeline = PipelineService(
            camera=self.camera, cv=self.cv, db=self.db,
            student_id=1, session_id=2,
        )

    def _result(self, faces: bool, pose: bool) -> CVProcessResult:
        """构造单帧视觉处理结果。"""
        eye = EyeState(confidence=1.0)
        head_pose = HeadPoseResult(confidence=1.0)
        mouth = MouthState(confidence=1.0)
        faces_list = [FaceLandmark(face_id=0, confidence=1.0)] if faces else []
        pose_landmark = PoseLandmark(confidence=1.0) if pose else None
        return CVProcessResult(
            faces=faces_list, eye=eye, head_pose=head_pose,
            mouth=mouth, pose=pose_landmark,
        )

    def test_track_behavior_writes_absent_event(self):
        """缺勤状态结束时应写入 BehaviorRecord。"""
        self.pipeline._track_behavior(self._result(faces=False, pose=False))
        self.assertEqual(len(self.db.behaviors), 0)
        self.pipeline._track_behavior(self._result(faces=True, pose=True))
        self.assertEqual(len(self.db.behaviors), 1)
        self.assertEqual(self.db.behaviors[0]["behavior_type"], "absent")


if __name__ == "__main__":
    unittest.main()
