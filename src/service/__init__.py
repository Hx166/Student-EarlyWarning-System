"""src.service —— 业务协调层。

负责将摄像头采集、视觉分析与数据存储串联为完整主流程。
"""

from __future__ import annotations

from typing import Any

__all__ = ["PipelineService", "PredictionService"]


def __getattr__(name: str) -> Any:
    """按需加载服务类，避免导入预测或回放时连带加载摄像头依赖。"""
    if name == "PipelineService":
        from src.service.pipeline import PipelineService

        return PipelineService
    if name == "PredictionService":
        from src.service.prediction import PredictionService

        return PredictionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
