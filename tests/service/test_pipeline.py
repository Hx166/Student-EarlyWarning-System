"""tests/service/test_pipeline.py \u2014\u2014 \u4e3b\u6d41\u7a0b\u670d\u52a1\u6d4b\u8bd5\u3002"""

from __future__ import annotations

import unittest
from unittest import mock

from src.core.models import (
    EyeState,
    FatigueResult,
    HeadPoseResult,
    EngagementResult,
    FaceLandmark,
)
from src.cv.processor import CVProcessResult
from src.service.pipeline import PipelineService


class _FakeDb:
    """\u7b80\u6613 mock \u6570\u636e\u5e93\uff0c\u8bb0\u5f55\u8c03\u7528\u3002"""

    def __init__(self):
        self.frame_records = []
        self.state_records = []
        self.warnings = []

    def insert_frame_record(self, **kwargs):
        self.frame_records.append(kwargs)
        return len(self.frame_records)

    def insert_state_record(self, **kwargs):
        self.state_records.append(kwargs)
        return len(self.state_records)

    def insert_warning(self, **kwargs):
        self.warnings.append(kwargs)
        return len(self.warnings)


class PipelineServiceTest(unittest.TestCase):
    """\u4e3b\u6d41\u7a0b\u670d\u52a1\u5355\u5143\u6d4b\u8bd5\u3002"""

    def setUp(self):
        """\u6784\u5efa mock \u7ec4\u4ef6\u4e0e\u7f16\u6392\u670d\u52a1\u3002"""
        self.camera = mock.MagicMock()
        self.cv = mock.MagicMock()
        self.db = _FakeDb()
        self.pipeline = PipelineService(
            camera=self.camera, cv=self.cv, db=self.db,
            student_id=1, session_id=2,
        )

    def _make_result(self, ear=0.3, pitch=5.0, yaw=0.0, roll=0.0,
                     perclos=0.1, fatigue=20.0, engagement=85.0, faces=False):
        """\u6784\u9020\u89c6\u89c9\u5904\u7406\u7ed3\u679c\u3002"""
        eye = EyeState(average_ear=ear, perclos=perclos, confidence=1.0)
        head = HeadPoseResult(pitch=pitch, yaw=yaw, roll=roll, confidence=1.0)
        fatigue_obj = FatigueResult(fatigue_score=fatigue, confidence=1.0)
        eng = EngagementResult(engagement_score=engagement, confidence=1.0)
        fl = [FaceLandmark(face_id=0, confidence=1.0)] if faces else []
        return CVProcessResult(faces=fl, eye=eye, head_pose=head,
                               fatigue=fatigue_obj, engagement=eng)

    def test_process_one_writes_frame_and_state(self):
        """\u5355\u5e27\u5904\u7406\u5199\u5165 FrameRecord\uff0cStateRecord \u5ef6\u65f6\u3002"""
        frame = mock.MagicMock()
        frame.frame_id = 1
        self.camera.read.return_value = frame
        self.cv.process.return_value = self._make_result()

        result = self.pipeline.process_one()
        self.assertIsNotNone(result)
        self.assertEqual(len(self.db.frame_records), 1)
        self.assertEqual(len(self.db.state_records), 0)

    def test_state_record_flush_after_window(self):
        """\u65f6\u95f4\u7a97\u53e3\u5230\u671f\u540e\u5199\u5165 StateRecord\u3002"""
        self.pipeline.frame_record_enabled = False
        self.camera.read.return_value = mock.MagicMock()
        self.cv.process.return_value = self._make_result(faces=True)
        seq = [0.0, 0.0, 2.0, 2.0]
        with mock.patch("src.service.pipeline.time.monotonic", side_effect=seq):
            self.pipeline.process_one()
            self.pipeline.process_one()
        self.assertEqual(len(self.db.state_records), 1)

    def test_warning_on_high_fatigue(self):
        """\u9ad8\u75b2\u52b3\u89e6\u53d1\u9884\u8b66\u3002"""
        self.pipeline._write_warning(warning_type="fatigue", warning_level="high",
                                    reason="high_fatigue_score", attention_score=30.0)
        self.assertEqual(len(self.db.warnings), 1)
        self.assertEqual(self.db.warnings[0]["warning_type"], "fatigue")

    def test_warning_cooldown_dedup(self):
        """\u540c\u7c7b\u578b\u9884\u8b66\u5728\u95f4\u9694\u5185\u53bb\u91cd\u3002"""
        self.camera.read.return_value = mock.MagicMock()
        res = self._make_result(fatigue=90.0)
        self.cv.process.return_value = res
        with mock.patch("src.service.pipeline.time.monotonic", return_value=0.0):
            self.pipeline._maybe_write_warning(res)
            self.pipeline._maybe_write_warning(res)
        fatigue_warnings = [w for w in self.db.warnings if w["warning_type"] == "fatigue"]
        self.assertEqual(len(fatigue_warnings), 1)

    def test_run_counts_frames(self):
        """run \u8fd4\u56de\u5904\u7406\u5e27\u6570\u3002"""
        self.pipeline.frame_record_enabled = False
        self.camera.read.side_effect = [mock.MagicMock(), mock.MagicMock(), None]
        self.cv.process.return_value = self._make_result()
        count = self.pipeline.run(max_frames=5)
        self.assertEqual(count, 2)
    def test_frame_record_disabled_via_config(self):
        """\u914d\u7f6e\u5173\u95ed FrameRecord \u65f6\u4e0d\u5199\u5165\u3002"""
        with mock.patch("src.service.pipeline.get_database", return_value={"frame_record": {"enabled": False}}):
            pipeline = PipelineService(
                camera=self.camera, cv=self.cv, db=self.db,
                student_id=1, session_id=2,
            )
        frame = mock.MagicMock()
        frame.frame_id = 1
        self.camera.read.return_value = frame
        self.cv.process.return_value = self._make_result()
        pipeline.process_one()
        self.assertEqual(len(self.db.frame_records), 0)



if __name__ == "__main__":
    unittest.main()
