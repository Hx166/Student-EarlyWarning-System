"""src/api/service/prediction.py —— API 预测服务单例。

为 Flask 预测接口提供模型注册表与风险预测器的惰性单例，
模型默认通过 ModelRegistry 从磁盘加载（models/xgboost/<version>），
缺失时由调用方降级处理。
"""

from __future__ import annotations

from typing import Optional

from src.config import get_system
from src.models import ModelRegistry
from src.models.risk_prediction import RiskPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

_model_registry: Optional[ModelRegistry] = None
_risk_predictor: Optional[RiskPredictor] = None


def _prediction_config() -> dict:
    """读取预测相关配置。

    Returns:
        预测配置字典。
    """
    return get_system().get("prediction", {})


def get_model_name() -> str:
    """获取预测使用的模型名称。

    Returns:
        模型名称字符串。
    """
    return str(_prediction_config().get("model_name", "xgboost"))


def get_model_version() -> Optional[str]:
    """获取预测使用的模型版本。

    Returns:
        模型版本号；未配置时返回 None（使用默认版本）。
    """
    return _prediction_config().get("model_version")


def get_model_registry() -> ModelRegistry:
    """获取模型注册表单例。

    Returns:
        ModelRegistry 实例。
    """
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def get_risk_predictor() -> RiskPredictor:
    """获取风险预测器单例。

    Returns:
        RiskPredictor 实例。
    """
    global _risk_predictor
    if _risk_predictor is None:
        _risk_predictor = RiskPredictor(
            registry=get_model_registry(),
            model_name=get_model_name(),
            model_version=get_model_version(),
        )
    return _risk_predictor


def reset() -> None:
    """重置单例（主要用于测试隔离）。"""
    global _model_registry, _risk_predictor
    _model_registry = None
    _risk_predictor = None
