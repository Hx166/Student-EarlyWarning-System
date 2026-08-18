"""tests/service/test_behavior_tracker.py —— 行为事件状态机测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from src.core.models import (
    EyeState,
    FaceLandmark,
    HeadPoseResult,
    MouthState,
    PoseLandmark,
)
from src.cv.processor import CVProcessResult
from src.service.behavior_tracker import BehaviorEventTracker


class BehaviorEventTrackerTest(unittest.TestCase):
    """行为事件跟踪器测试。"""

    def setUp(self):
        """初始化跟踪器与基准时间。"""
        self.tracker = BehaviorEventTracker()
        self.t0 = datetime(2026, 8, 9, 9, 0, 0)

    def _result(
        self,
        faces: bool = True,
        pose: bool = True,
        head: str = "Normal",
        blink: bool = False,
        yawn: bool = False,
    ) -> CVProcessResult:
        """构造单帧视觉处理结果。"""
        eye = EyeState(blink=blink, confidence=1.0)
        head_pose = HeadPoseResult(pose_status=head, confidence=1.0)
        mouth = MouthState(is_yawn=yawn, confidence=1.0)
        faces_list = [FaceLandmark(face_id=0, confidence=1.0)] if faces else []
        pose_landmark = PoseLandmark(confidence=1.0) if pose else None
        return CVProcessResult(
            faces=faces_list, eye=eye, head_pose=head_pose,
            mouth=mouth, pose=pose_landmark,
        )

    def test_absent_event_start_end(self):
        """缺勤事件应在状态结束时生成并包含时长。"""
        self.tracker.update(self._result(faces=False, pose=False), self.t0)
        events = self.tracker.update(
            self._result(faces=True, pose=True), self.t0 + timedelta(seconds=5)
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].behavior_type, "absent")
        self.assertAlmostEqual(events[0].duration, 5.0, places=6)

    def test_looking_away_when_face_lost_but_pose_present(self):
        """有人体姿态但无人脸时应判定为转头而非缺勤。"""
        self.tracker.update(self._result(faces=True, pose=True), self.t0)
        self.tracker.update(
            self._result(faces=False, pose=True), self.t0 + timedelta(seconds=3)
        )
        events = self.tracker.update(
            self._result(faces=True, pose=True), self.t0 + timedelta(seconds=6)
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].behavior_type, "looking_away")
        self.assertAlmostEqual(events[0].duration, 3.0, places=6)

    def test_head_down_event(self):
        """低头事件在状态结束时生成。"""
        self.tracker.update(self._result(head="Normal"), self.t0)
        self.tracker.update(self._result(head="LookDown"), self.t0 + timedelta(seconds=2))
        events = self.tracker.update(self._result(head="Normal"), self.t0 + timedelta(seconds=5))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].behavior_type, "head_down")
        self.assertAlmostEqual(events[0].duration, 3.0, places=6)

    def test_yawn_event(self):
        """打哈欠事件在状态结束时生成。"""
        self.tracker.update(self._result(yawn=False), self.t0)
        self.tracker.update(self._result(yawn=True), self.t0 + timedelta(seconds=1))
        events = self.tracker.update(self._result(yawn=False), self.t0 + timedelta(seconds=4))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].behavior_type, "yawn")
        self.assertAlmostEqual(events[0].duration, 3.0, places=6)

    def test_blink_rising_edge(self):
        """眨眼按上升沿生成单次事件。"""
        self.tracker.update(self._result(blink=False), self.t0)
        events = self.tracker.update(self._result(blink=True), self.t0 + timedelta(seconds=1))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].behavior_type, "blink")
        events = self.tracker.update(self._result(blink=True), self.t0 + timedelta(seconds=2))
        self.assertEqual(events, [])

    def test_flush_active_events(self):
        """flush 应结束仍在活跃中的事件。"""
        self.tracker.update(self._result(faces=False, pose=False), self.t0)
        events = self.tracker.flush(self.t0 + timedelta(seconds=10))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].behavior_type, "absent")
        self.assertAlmostEqual(events[0].duration, 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
