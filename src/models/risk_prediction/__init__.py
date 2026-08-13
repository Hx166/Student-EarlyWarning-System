"""src.models.risk_prediction —— 风险预测/推理模块。

将 Feature Engineering 输出接入由 Model Management 加载的
XGBoost 模型，输出统一的 RiskPredictionResult。
"""

from src.core.models import RiskPredictionResult
from src.models.risk_prediction.predictor import RiskPredictor

__all__ = ["RiskPredictor", "RiskPredictionResult"]
