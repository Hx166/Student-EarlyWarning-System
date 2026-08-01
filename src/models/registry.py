"""src/models/registry.py —— 模型注册表。

统一管理机器学习模型的注册与访问接口。
第四阶段实现具体模型，此处提供框架。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """模型注册表。

    Attributes:
        _models: 已注册模型字典。
    """

    def __init__(self) -> None:
        """初始化模型注册表。"""
        self._models: Dict[str, object] = {}
        logger.info("ModelRegistry 已初始化")

    def register(self, name: str, model: object) -> None:
        """注册一个模型。

        Args:
            name: 模型名称。
            model: 模型实例。
        """
        self._models[name] = model
        logger.info("已注册模型: %s", name)

    def get(self, name: str) -> Optional[object]:
        """获取指定模型。

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
