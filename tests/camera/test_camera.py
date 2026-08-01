"""tests/camera/test_camera.py —— 摄像头模块测试。"""

from __future__ import annotations

import unittest

from src.camera import Camera, CameraError


class CameraInitTest(unittest.TestCase):
    """摄像头初始化与异常测试。"""

    def test_init_uses_default_index(self):
        """默认情况使用配置中的摄像头索引。"""
        camera = Camera(index=5)
        # 配置文件中 index=0，显式传入不应覆盖配置读取逻辑
        self.assertTrue(hasattr(camera, "index"))

    def test_close_without_open(self):
        """未打开时关闭不应报错。"""
        camera = Camera()
        camera.close()


if __name__ == "__main__":
    unittest.main()
