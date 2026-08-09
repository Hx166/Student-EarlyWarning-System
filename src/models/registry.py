"""src/models/registry.py —— 模型注册与管理。

统一管理机器学习模型的版本、元数据与访问入口：
- 内存注册（register/get/list_models）；
- 版本化目录保存（models/<name>/<version>）与加载；
- 元数据（version/feature_names/config/saved_at）读写；
- 模型状态检查（check_model）；
- 为预测/推理模块提供统一模型获取入口（get_model）。

复用具体模型（如 XGBoostModel）自身的 save/load 能力，
本模块只负责组织目录、版本与元数据，不重复实现模型算法。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 默认模型根目录（项目根/models）
MODELS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
)

# 模型元数据文件名（保存于各版本目录内）
METADATA_FILENAME = "model_metadata.json"

# 默认版本号
DEFAULT_VERSION = "v1"

# 支持的模型类型与加载工厂
_MODEL_LOADERS: Dict[str, Any] = {}


def _resolve_models_root(models_root: Optional[str]) -> str:
    """解析模型根目录。

    Args:
        models_root: 显式根目录（可空，使用项目默认 models/）。

    Returns:
        模型根目录绝对路径。
    """
    if models_root is None:
        return MODELS_ROOT
    return os.path.abspath(models_root)


def register_model_loader(model_type: str, factory: Any) -> None:
    """注册模型类型对应的加载工厂。

    Args:
        model_type: 模型类型标识（如 xgboost）。
        factory: 无参可调用，返回新模型实例（实现 load(dir)）。
    """
    _MODEL_LOADERS[model_type] = factory
    logger.info("已注册模型加载工厂: %s", model_type)


def _model_dir(models_root: str, name: str, version: str) -> str:
    """构造模型版本目录路径。

    Args:
        models_root: 模型根目录。
        name: 模型名称。
        version: 版本号。

    Returns:
        版本目录绝对路径。
    """
    return os.path.join(models_root, name, version)


class ModelRegistry:
    """模型注册与管理器。

    Attributes:
        models_root: 模型根目录（默认项目 models/）。
        _models: 内存中已注册/已加载的模型实例缓存。
    """

    def __init__(self, models_root: Optional[str] = None) -> None:
        """初始化模型注册表。

        Args:
            models_root: 模型根目录；为空时使用项目默认 models/。
        """
        self.models_root: str = _resolve_models_root(models_root)
        self._models: Dict[str, Any] = {}
        os.makedirs(self.models_root, exist_ok=True)
        logger.info("ModelRegistry 已初始化，根目录=%s", self.models_root)

    def register(self, name: str, model: object) -> None:
        """注册一个模型到内存。

        Args:
            name: 模型名称（唯一标识）。
            model: 模型实例。
        """
        self._models[name] = model
        logger.info("已注册模型: %s", name)

    def get(self, name: str) -> Optional[object]:
        """获取已注册模型。

        Args:
            name: 模型名称。

        Returns:
            模型实例；未注册时返回 None。
        """
        return self._models.get(name)

    def list_models(self) -> list:
        """列出所有已注册模型名称。

        Returns:
            模型名称列表。
        """
        return list(self._models.keys())

    def _cache_key(self, name: str, version: str) -> str:
        """构造缓存键。

        Args:
            name: 模型名称。
            version: 版本号。

        Returns:
            缓存键（name@version）。
        """
        return f"{name}@{version}"

    def save_model(
        self,
        name: str,
        model: object,
        version: str = DEFAULT_VERSION,
        model_type: str = "xgboost",
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """保存模型到版本化目录并写入元数据。

        复用模型自身的 save(dir) 接口完成模型文件落盘。

        Args:
            name: 模型名称。
            model: 实现了 save(dir) 的模型实例（如 XGBoostModel）。
            version: 版本号（默认 v1）。
            model_type: 模型类型标识（默认 xgboost）。
            feature_names: 特征列名列表（可空，由模型元数据补齐）。
            metadata: 额外元数据（可空）。

        Returns:
            保存的版本目录路径。

        Raises:
            ValueError: 模型缺少 save 接口。
        """
        if not hasattr(model, "save"):
            raise ValueError("模型必须实现 save(dir) 接口")
        target_dir = _model_dir(self.models_root, name, version)
        model.save(target_dir)

        feature_names = feature_names or getattr(model, "feature_names", None) or []
        model_metadata: Dict[str, Any] = {
            "name": name,
            "version": version,
            "model_type": model_type,
            "feature_names": list(feature_names) if feature_names else [],
            "saved_at": datetime.now().isoformat(),
        }
        model_metadata.update(metadata or {})
        with open(
            os.path.join(target_dir, METADATA_FILENAME), "w", encoding="utf-8"
        ) as fh:
            json.dump(model_metadata, fh, ensure_ascii=False, indent=2)

        self._models[self._cache_key(name, version)] = model
        logger.info("模型已保存: %s@%s -> %s", name, version, target_dir)
        return target_dir

    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        model_factory: Optional[Any] = None,
    ) -> object:
        """从磁盘加载指定版本模型。

        Args:
            name: 模型名称。
            version: 版本号（为空时取默认版本 v1）。
            model_factory: 模型工厂（无参可调用，返回新模型实例）。
                为空时按元数据 model_type 选择默认工厂。

        Returns:
            加载后的模型实例（已缓存到注册表）。

        Raises:
            FileNotFoundError: 模型目录或元数据不存在。
            ValueError: 模型类型不受支持且未提供 factory。
        """
        version = version or DEFAULT_VERSION
        target_dir = _model_dir(self.models_root, name, version)
        metadata_path = os.path.join(target_dir, METADATA_FILENAME)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"模型元数据不存在: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as fh:
            model_metadata = json.load(fh)
        model_type = model_metadata.get("model_type", "xgboost")

        factory = model_factory or _MODEL_LOADERS.get(model_type)
        if factory is None:
            raise ValueError(f"不支持的模型类型: {model_type}，请提供 model_factory")
        model = factory()
        model.load(target_dir)

        cache_key = self._cache_key(name, version)
        self._models[cache_key] = model
        if name not in self._models:
            self._models[name] = model
        logger.info("模型已加载: %s@%s", name, version)
        return model

    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        model_factory: Optional[Any] = None,
    ) -> Optional[object]:
        """统一模型获取入口。

        优先级：内存注册 → 已加载缓存 → 尝试从磁盘加载。

        Args:
            name: 模型名称。
            version: 版本号（为空时取默认版本）。
            model_factory: 加载时使用的模型工厂（可空）。

        Returns:
            模型实例；不存在时返回 None。
        """
        if name in self._models:
            return self._models[name]
        version = version or DEFAULT_VERSION
        cache_key = self._cache_key(name, version)
        if cache_key in self._models:
            return self._models[cache_key]
        try:
            return self.load_model(name, version, model_factory)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("模型获取失败: %s@%s (%s)", name, version, exc)
            return None

    def list_versions(self, name: str) -> List[str]:
        """列出某模型的所有版本。

        Args:
            name: 模型名称。

        Returns:
            版本号列表（已排序）。
        """
        root = os.path.join(self.models_root, name)
        if not os.path.isdir(root):
            return []
        return sorted(
            entry for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry))
        )

    def get_metadata(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """读取指定版本模型的元数据。

        Args:
            name: 模型名称。
            version: 版本号（为空时取默认版本）。

        Returns:
            元数据字典。

        Raises:
            FileNotFoundError: 元数据文件不存在。
        """
        version = version or DEFAULT_VERSION
        metadata_path = os.path.join(
            self.models_root, name, version, METADATA_FILENAME
        )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"模型元数据不存在: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def check_model(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """检查模型状态。

        返回状态字典，包含：是否存在、是否已加载、
        版本目录、元数据与特征列名。

        Args:
            name: 模型名称。
            version: 版本号（为空时取默认版本）。

        Returns:
            状态字典。
        """
        version = version or DEFAULT_VERSION
        target_dir = _model_dir(self.models_root, name, version)
        metadata_path = os.path.join(target_dir, METADATA_FILENAME)
        exists = os.path.exists(metadata_path)
        cache_key = self._cache_key(name, version)
        loaded = cache_key in self._models or name in self._models
        status: Dict[str, Any] = {
            "name": name,
            "version": version,
            "exists": exists,
            "loaded": loaded,
            "path": target_dir,
        }
        if exists:
            try:
                status["metadata"] = self.get_metadata(name, version)
                status["feature_names"] = status["metadata"].get(
                    "feature_names", []
                )
            except FileNotFoundError:
                status["metadata"] = None
                status["feature_names"] = []
        return status

    def remove_model(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> bool:
        """删除指定版本模型（目录与缓存）。

        Args:
            name: 模型名称。
            version: 版本号（为空时取默认版本）。

        Returns:
            是否实际删除。
        """
        version = version or DEFAULT_VERSION
        target_dir = _model_dir(self.models_root, name, version)
        cache_key = self._cache_key(name, version)
        self._models.pop(cache_key, None)
        if not os.path.isdir(target_dir):
            logger.warning("待删除模型目录不存在: %s", target_dir)
            return False
        import shutil

        shutil.rmtree(target_dir)
        logger.info("模型已删除: %s@%s", name, version)
        return True
