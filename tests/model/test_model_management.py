"""tests/model/test_model_management.py —— 模型管理模块测试。

覆盖模型注册、版本化保存/加载、元数据、状态检查、
异常路径以及与 XGBoostModel 保存能力的集成。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, List

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.registry import (
    METADATA_FILENAME,
    ModelRegistry,
    register_model_loader,
)
from src.models.xgboost import XGBoostModel


class FakeBackend:
    """模拟模型算法后端（save/load 落盘）。"""

    def __init__(self, marker: str = "x") -> None:
        """初始化桩后端。

        Args:
            marker: 模型文件内容标记。
        """
        self.marker = marker

    def save_model(self, path: str) -> None:
        """写模型文件。"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"model-{self.marker}")

    def load_model(self, path: str) -> None:
        """读模型文件。"""
        with open(path, "r", encoding="utf-8") as fh:
            fh.read()


def make_xgboost_model(marker: str = "x") -> XGBoostModel:
    """构造可保存（已训练）的 XGBoostModel 实例。

    Args:
        marker: 后端标记。

    Returns:
        带桩后端且 is_trained=True 的 XGBoostModel。
    """
    model = XGBoostModel()
    model.model = FakeBackend(marker)
    model.feature_names = list(FEATURE_VECTOR_COLUMNS)
    model.is_trained = True
    return model


class FakeLoadableModel:
    """实现 save(dir)/load(dir) 接口的桩模型（用于加载路径测试）。"""

    def __init__(self, name: str = "") -> None:
        """初始化桩模型。"""
        self.name = name
        self.feature_names = list(FEATURE_VECTOR_COLUMNS)

    def save(self, model_dir: str) -> None:
        """桩保存。"""
        os.makedirs(model_dir, exist_ok=True)

    def load(self, model_dir: str) -> "FakeLoadableModel":
        """桩加载。"""
        self.name = "loaded-from-disk"
        return self


class ModelRegistryTest(unittest.TestCase):
    """模型注册表兼容行为测试。"""

    def setUp(self) -> None:
        """创建临时根目录。"""
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.registry = ModelRegistry(models_root=self.root)

    def tearDown(self) -> None:
        """清理临时目录。"""
        self._tmp.cleanup()

    def test_register_and_get(self):
        """注册后可通过名称获取。"""
        model = object()
        self.registry.register("xgboost", model)
        self.assertIs(self.registry.get("xgboost"), model)

    def test_list_models(self):
        """应列出已注册模型。"""
        self.registry.register("m1", object())
        self.registry.register("m2", object())
        self.assertEqual(sorted(self.registry.list_models()), ["m1", "m2"])

    def test_get_missing_returns_none(self):
        """未注册模型返回 None。"""
        self.assertIsNone(self.registry.get("not_exist"))


class ModelManagementTest(unittest.TestCase):
    """模型管理（保存/加载/元数据/状态）测试。"""

    def setUp(self) -> None:
        """创建临时根目录。"""
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.registry = ModelRegistry(models_root=self.root)

    def tearDown(self) -> None:
        """清理临时目录。"""
        self._tmp.cleanup()

    def test_save_model_writes_files_and_metadata(self):
        """保存应写入模型文件、元数据与特征列。"""
        model = make_xgboost_model()
        target = self.registry.save_model("xgboost", model, version="v1")
        self.assertTrue(os.path.isdir(target))
        for filename in (
            "xgboost_model.json",
            "feature_names.json",
            "label_mapping.json",
            "training_config.yaml",
            METADATA_FILENAME,
        ):
            self.assertTrue(
                os.path.exists(os.path.join(target, filename)),
                f"缺少文件: {filename}",
            )
        with open(os.path.join(target, METADATA_FILENAME), "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        self.assertEqual(metadata["name"], "xgboost")
        self.assertEqual(metadata["version"], "v1")
        self.assertEqual(metadata["model_type"], "xgboost")
        self.assertEqual(metadata["feature_names"], FEATURE_VECTOR_COLUMNS)

    def test_save_model_without_save_raises(self):
        """缺少 save 接口的模型应抛 ValueError。"""
        with self.assertRaises(ValueError):
            self.registry.save_model("bad", object())

    def test_load_model_with_factory(self):
        """使用模型工厂加载应返回缓存实例。"""
        self.registry.save_model(
            "demo", FakeLoadableModel("disk"), version="v1", model_type="demo"
        )
        loaded = self.registry.load_model(
            "demo", version="v1", model_factory=lambda: FakeLoadableModel()
        )
        self.assertEqual(loaded.name, "loaded-from-disk")
        self.assertIs(
            self.registry.get_model("demo", version="v1"),
            loaded,
        )

    def test_load_model_missing_raises(self):
        """模型不存在时 load_model 应抛 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            self.registry.load_model("missing", version="v1")

    def test_get_model_missing_returns_none(self):
        """get_model 对不存在模型应返回 None。"""
        self.assertIsNone(self.registry.get_model("missing"))

    def test_list_versions(self):
        """应列出模型的所有版本。"""
        self.registry.save_model("m", FakeLoadableModel(), version="v1")
        self.registry.save_model("m", FakeLoadableModel(), version="v2")
        self.assertEqual(self.registry.list_versions("m"), ["v1", "v2"])
        self.assertEqual(self.registry.list_versions("not_exist"), [])

    def test_get_metadata_missing_raises(self):
        """元数据不存在时应抛 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            self.registry.get_metadata("not_exist")

    def test_check_model_status(self):
        """状态检查应正确反映存在/加载/特征列。"""
        model = make_xgboost_model()
        self.registry.save_model("xgboost", model, version="v1")
        status = self.registry.check_model("xgboost", version="v1")
        self.assertTrue(status["exists"])
        self.assertTrue(status["loaded"])
        self.assertEqual(status["feature_names"], FEATURE_VECTOR_COLUMNS)
        missing = self.registry.check_model("ghost")
        self.assertFalse(missing["exists"])
        self.assertFalse(missing["loaded"])

    def test_check_model_not_loaded(self):
        """未加载的已保存模型 loaded 应反映真实缓存状态。"""
        # 保存到磁盘但不保留缓存引用（新 registry 实例）
        registry2 = ModelRegistry(models_root=self.root)
        registry2.save_model("m", FakeLoadableModel(), version="v1")
        fresh = ModelRegistry(models_root=self.root)
        status = fresh.check_model("m", version="v1")
        self.assertTrue(status["exists"])
        self.assertFalse(status["loaded"])

    def test_get_model_fallback_to_disk(self):
        """get_model 在未注册时应从磁盘加载。"""
        self.registry.save_model(
            "demo", FakeLoadableModel("disk"), version="v1", model_type="demo"
        )
        fresh = ModelRegistry(models_root=self.root)
        loaded = fresh.get_model(
            "demo", version="v1", model_factory=lambda: FakeLoadableModel()
        )
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "loaded-from-disk")

    def test_remove_model(self):
        """删除模型应移除目录并返回 True。"""
        target = self.registry.save_model(
            "m", FakeLoadableModel(), version="v1"
        )
        self.assertTrue(self.registry.remove_model("m", version="v1"))
        self.assertFalse(os.path.exists(target))
        self.assertFalse(self.registry.remove_model("m", version="v1"))

    def test_unsupported_type_without_factory(self):
        """未知模型类型且无 factory 时 load 应抛 ValueError。"""
        self.registry.save_model(
            "demo", FakeLoadableModel(), version="v1", model_type="unknown"
        )
        with self.assertRaises(ValueError):
            self.registry.load_model("demo", version="v1")


class XGBoostIntegrationTest(unittest.TestCase):
    """与 XGBoostModel 的集成测试。"""

    def setUp(self) -> None:
        """创建临时根目录并注册 xgboost 工厂。"""
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.registry = ModelRegistry(models_root=self.root)
        register_model_loader("xgboost", lambda: XGBoostModel())

    def tearDown(self) -> None:
        """清理临时目录。"""
        self._tmp.cleanup()

    def test_save_xgboost_and_read_metadata(self):
        """通过 registry 保存 XGBoost 模型并读取元数据。"""
        model = make_xgboost_model(marker="xgb")
        self.registry.save_model("xgboost", model, version="v1")
        metadata = self.registry.get_metadata("xgboost", version="v1")
        self.assertEqual(metadata["model_type"], "xgboost")
        self.assertEqual(metadata["feature_names"], FEATURE_VECTOR_COLUMNS)
        with open(
            os.path.join(self.root, "xgboost", "v1", "xgboost_model.json"),
            "r",
            encoding="utf-8",
        ) as fh:
            self.assertEqual(fh.read(), "model-xgb")

    def test_get_model_registered_instance(self):
        """已注册实例应直接返回。"""
        model = make_xgboost_model()
        self.registry.register("xgboost", model)
        self.assertIs(self.registry.get_model("xgboost"), model)


if __name__ == "__main__":
    unittest.main()
