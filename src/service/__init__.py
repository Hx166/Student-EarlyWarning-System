"""src.service —— 业务协调层。

负责将摄像头采集、视觉分析与数据存储串联为完整主流程。
"""

from src.service.pipeline import PipelineService
from src.service.prediction import PredictionService

__all__ = ["PipelineService", "PredictionService"]
