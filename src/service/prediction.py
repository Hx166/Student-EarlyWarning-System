"""src/service/prediction.py —— 实时预测接线模块。

将第四阶段独立的 Feature Engineering 与 Risk Prediction 接入运行时：
- 维护 StateRecord 滑动窗口缓冲；
- 累积到足够记录后用 FeatureBuilder 构建特征向量；
- 调用 RiskPredictor 得到 RiskPredictionResult。

本模块不修改 FeatureBuilder / RiskPredictor 的公开接口，
仅在其上层提供接线逻辑。模型一律通过 ModelRegistry 统一获取。
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence

from src.core.models import RiskPredictionResult
from src.models import ModelRegistry
from src.models.features.builder import FeatureBuilder
from src.models.labels import ENGAGEMENT_LABEL_MAPPING
from src.models.risk_prediction import RiskPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 默认模型名称（与 src/models/__init__.py 注册一致）
_DEFAULT_MODEL_NAME = "xgboost"
# 默认特征窗口长度（秒）
_DEFAULT_WINDOW_SECONDS = 30
# 默认触发预测所需的最少记录数（约等于窗口长度，1 条/秒）
_DEFAULT_MIN_RECORDS = 30


class PredictionService:
    """实时预测接线服务。

    维护 StateRecord 滑动窗口缓冲，当累积到足够记录时构建特征向量，
    调用 RiskPredictor 输出 RiskPredictionResult。

    Attributes:
        model_name: 模型名称。
        window_seconds: 特征窗口长度（秒）。
        min_records: 触发预测所需的最小记录数。
        _builder: 特征向量构建器。
        _buffer: 状态记录滑动窗口缓冲。
        _registry: 模型注册表（延迟创建）。
        _predictor: 风险预测器（延迟创建）。
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        window_seconds: int = _DEFAULT_WINDOW_SECONDS,
        min_records: int = _DEFAULT_MIN_RECORDS,
        registry: Optional[ModelRegistry] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """初始化实时预测接线服务。

        Args:
            model_name: 模型名称（默认 xgboost）。
            window_seconds: 特征窗口长度（秒）。
            min_records: 触发预测所需的最小记录数。
            registry: 可选的模型注册表（测试注入用），为空时延迟创建。
            model_version: 模型版本；为空时使用注册表默认版本。
        """
        self.model_name = model_name
        self.window_seconds = window_seconds
        self.min_records = min_records
        self.model_version = model_version
        self._builder = FeatureBuilder(window_seconds=window_seconds)
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=window_seconds)
        self._registry = registry
        self._predictor: Optional[RiskPredictor] = None
        self._last_probabilities: Dict[str, float] = {}
        self._behavior_records: list[Dict[str, Any]] = []
        logger.info(
            "PredictionService 已初始化，model=%s window=%s秒 min_records=%s",
            model_name, window_seconds, min_records,
        )

    def predict(
        self,
        state_record: Dict[str, Any],
        behavior_records: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[RiskPredictionResult]:
        """累积一条状态记录并返回预测结果。

        Args:
            state_record: 单条 StateRecord 字段字典。
            behavior_records: 当前会话已完成的行为事件（可空）。

        Returns:
            风险预测结果；数据不足、模型不可用或预测异常时返回 None。
        """
        self._buffer.append(state_record)
        if behavior_records is not None:
            self._behavior_records = list(behavior_records)
        if len(self._buffer) < self.min_records:
            return None

        try:
            features = self._builder.build(
                list(self._buffer), self._behavior_records or None
            )
            if features.empty:
                return None
            latest = features.iloc[[-1]]
            predictor = self._ensure_predictor()
            result, probabilities = predictor.predict_with_proba(latest)
            self._last_probabilities = {
                ENGAGEMENT_LABEL_MAPPING[index]: float(probabilities[index])
                for index in range(4)
            }
            return result
        except (RuntimeError, ValueError) as exc:
            logger.warning("预测跳过，模型或特征异常: %s", exc)
            return None
        except Exception as exc:  # 容错：单次预测异常不影响主流程
            logger.warning("预测异常，跳过本轮预测: %s", exc)
            return None

    def _ensure_predictor(self) -> RiskPredictor:
        """延迟创建风险预测器。

        Returns:
            已初始化的风险预测器。
        """
        if self._predictor is None:
            if self._registry is None:
                self._registry = ModelRegistry()
            self._predictor = RiskPredictor(
                registry=self._registry,
                model_name=self.model_name,
                model_version=self.model_version,
            )
        return self._predictor

    @property
    def last_probabilities(self) -> Dict[str, float]:
        """返回最近一次预测的四类概率快照。"""
        return dict(self._last_probabilities)
