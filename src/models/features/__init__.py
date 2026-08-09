"""src.models.features —— 特征工程模块。

提供状态记录与行为记录到机器学习特征向量的构建能力，
供 XGBoost / LSTM 等模型训练与预测使用。
"""

from src.models.features.builder import FeatureBuilder
from src.models.features.definitions import (
    FEATURE_VECTOR_COLUMNS,
    STATISTICAL_FEATURE_COLUMNS,
)

__all__ = [
    "FeatureBuilder",
    "FEATURE_VECTOR_COLUMNS",
    "STATISTICAL_FEATURE_COLUMNS",
]
