"""tests/service/test_pipeline_fps.py —— 后端精确 FPS 统计测试。"""

from __future__ import annotations

import unittest
from unittest import mock

from src.service.pipeline import PipelineService


class _FakeDb:
    """记录调用的数据库桩。"""

    def insert_behavior(self, **kwargs):
        return 0


class PipelineFpsTest(unittest.TestCase):
    """Pipeline FPS 统计测试。"""

    def setUp(self):
        """构建 pipeline。"""
        self.camera = mock.MagicMock()
        self.cv = mock.MagicMock()
        self.db = _FakeDb()
        self.pipeline = PipelineService(
            camera=self.camera, cv=self.cv, db=self.db,
            student_id=1, session_id=2,
        )

    def test_initial_fps_zero(self):
        """初始 FPS 应为 0。"""
        self.assertEqual(self.pipeline.current_fps, 0.0)

    def test_update_fps_after_window(self):
        """窗口到期后应计算 FPS。"""
        seq = [0.0, 0.0, 0.0, 2.0]
        with mock.patch("src.service.pipeline.time.monotonic", side_effect=seq):
            self.pipeline._update_fps()
            self.pipeline._update_fps()
            self.pipeline._update_fps()
            self.pipeline._update_fps()
        self.assertAlmostEqual(self.pipeline.current_fps, 2.0, places=6)


class FpsApiTest(unittest.TestCase):
    """FPS 查询接口测试。"""

    def test_fps_endpoint_returns_backend_fps(self):
        """接口应返回注入 pipeline 的 FPS。"""
        from src.api.app import create_app

        class _StubPipeline:
            current_fps = 12.5

        app = create_app(pipeline=_StubPipeline())
        app.config["TESTING"] = True
        client = app.test_client()
        res = client.get("/api/metrics/fps")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["code"], 0)
        self.assertAlmostEqual(payload["data"]["fps"], 12.5, places=2)


if __name__ == "__main__":
    unittest.main()
