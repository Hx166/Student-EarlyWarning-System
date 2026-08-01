"""tests/model/test_models.py —— 模型注册表测试。"""

from __future__ import annotations

import unittest

from src.models import ModelRegistry


class ModelRegistryTest(unittest.TestCase):
    """模型注册表测试。"""

    def test_register_and_get(self):
        """注册后可通过名称获取。"""
        registry = ModelRegistry()
        registry.register("xgboost", object())
        self.assertIsNotNone(registry.get("xgboost"))

    def test_get_missing_returns_none(self):
        """未注册模型返回 None。"""
        registry = ModelRegistry()
        self.assertIsNone(registry.get("not_exist"))


if __name__ == "__main__":
    unittest.main()
