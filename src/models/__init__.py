"""src.models —— 机器学习模型包（第四阶段）。

提供 XGBoost 模型、特征工程与统一模型管理入口。
"""

from src.models.registry import (
    METADATA_FILENAME,
    ModelRegistry,
    register_model_loader,
)
from src.models.xgboost import XGBoostModel

# 注册 xgboost 模型加载工厂，供 ModelRegistry.load_model 使用
register_model_loader("xgboost", lambda: XGBoostModel())

__all__ = ["ModelRegistry", "XGBoostModel", "register_model_loader", "METADATA_FILENAME"]
