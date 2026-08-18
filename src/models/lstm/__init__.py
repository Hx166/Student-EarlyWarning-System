"""src.models.lstm —— LSTM 时序预测模型（第五阶段实现）。

提供基于 PyTorch 的课堂状态趋势预测能力，
输出统一的 LSTMResult，通过 ModelRegistry 统一管理。
"""

from src.models.lstm.model import DEFAULT_PARAMS, LABEL_MAPPING, LSTMModel

__all__ = ["LSTMModel", "DEFAULT_PARAMS", "LABEL_MAPPING"]
