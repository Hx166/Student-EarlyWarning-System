"""tests/test_config.py —— 配置加载模块测试。"""

from __future__ import annotations

import unittest

from src.utils.config import clear_cache, get_database, get_system, load_config


class ConfigTest(unittest.TestCase):
    """配置加载测试。"""

    def setUp(self):
        """清空配置缓存。"""
        clear_cache()

    def tearDown(self):
        """清空配置缓存。"""
        clear_cache()

    def test_load_system(self):
        """系统配置可加载且包含 server 段。"""
        cfg = load_config("system")
        self.assertIn("system", cfg)
        self.assertIn("server", cfg)

    def test_get_system(self):
        """获取系统配置。"""
        self.assertIn("server", get_system())

    def test_get_database(self):
        """数据库配置包含 database 段。"""
        self.assertIn("database", get_database())


if __name__ == "__main__":
    unittest.main()
