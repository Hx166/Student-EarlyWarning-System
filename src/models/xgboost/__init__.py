"""src.models.xgboost —— XGBoost 课堂状态分类模型（第四阶段实现）。

提供 4 类状态分类（正常/注意力下降/疲劳/高风险）的训练、
预测、风险等级映射、模型保存/加载与评估能力。
"""

from src.models.xgboost.model import (
    DEFAULT_PARAMS,
    LABEL_MAPPING,
    LABEL_NAMES,
    RISK_THRESHOLDS,
    XGBoostModel,
)

__all__ = [
    "XGBoostModel",
    "DEFAULT_PARAMS",
    "LABEL_MAPPING",
    "LABEL_NAMES",
    "RISK_THRESHOLDS",
]
