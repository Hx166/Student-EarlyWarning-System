"""src/models/risk_prediction/predictor.py —— 风险预测/推理模块。

利用已经完成的 Feature Engineering 输出与 Model Management 加载的
XGBoost 模型，对新的特征数据进行风险预测，输出统一的 RiskPredictionResult。

本模块不重新设计特征、不重新训练模型、不直接读取模型文件，
模型一律通过 ModelRegistry.get_model 获取。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.models import RiskPredictionResult
from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskPredictor:
    """学业风险预测器。

    连接 Feature Engineering -> Model Management/XGBoost -> Risk Result，
    对特征数据进行风险预测。

    Attributes:
        registry: 模型管理器（用于获取模型）。
        model_name: 用于预测的模型名称。
        model_version: 使用的模型版本（为空取默认版本）。
        model_factory: 加载模型时的工厂（可空，使用默认 xgboost 工厂）。
    """

    def __init__(
        self,
        registry: ModelRegistry,
        model_name: str = "xgboost",
        model_version: Optional[str] = None,
        model_factory: Optional[Any] = None,
    ) -> None:
        """初始化风险预测器。

        Args:
            registry: 模型管理器实例。
            model_name: 模型名称（默认 xgboost）。
            model_version: 模型版本（为空用默认版本）。
            model_factory: 模型加载工厂（可空）。
        """
        self.registry = registry
        self.model_name = model_name
        self.model_version = model_version
        self.model_factory = model_factory
        self._model: Optional[Any] = None
        logger.info(
            "RiskPredictor 已初始化: %s@%s", model_name, model_version or "(default)"
        )

    def predict(self, features: pd.DataFrame) -> RiskPredictionResult:
        """对单行特征进行风险预测。

        Args:
            features: Feature Engineering 输出的特征 DataFrame（单行）。

        Returns:
            风险预测结果对象。

        Raises:
            ValueError: 输入为空、缺列、包含 label 或类型不正确。
            RuntimeError: 模型不存在或未加载。
        """
        if features is None or len(features) != 1:
            raise ValueError("RiskPredictor.predict 要求单行特征数据")
        result, _ = self.predict_with_proba(features)
        return result

    def predict_with_proba(
        self, features: pd.DataFrame
    ) -> tuple[RiskPredictionResult, np.ndarray]:
        """对单行特征预测，并同时返回四类概率。

        Args:
            features: Feature Engineering 输出的单行特征 DataFrame。

        Returns:
            单行预测结果与形状为 ``(4,)`` 的概率数组。

        Raises:
            ValueError: 输入为空、缺列、包含 label 或不是单行。
            RuntimeError: 模型不存在或未加载。
        """
        if features is None or len(features) != 1:
            raise ValueError("RiskPredictor.predict_with_proba 要求单行特征数据")
        results, probabilities = self._predict_batch_with_proba(features)
        return results[0], probabilities[0]

    def predict_batch(self, features: pd.DataFrame) -> List[RiskPredictionResult]:
        """对多行特征进行批量风险预测。

        Args:
            features: Feature Engineering 输出的特征 DataFrame（多行）。

        Returns:
            风险预测结果对象列表（每行一个）。

        Raises:
            ValueError: 输入为空、缺列、包含 label 或类型不正确。
            RuntimeError: 模型不存在或未加载。
        """
        results, _ = self._predict_batch_with_proba(features)
        return results

    def _predict_batch_with_proba(
        self, features: pd.DataFrame
    ) -> tuple[List[RiskPredictionResult], np.ndarray]:
        """批量预测并返回结果及概率矩阵。"""
        X_df = self._validate_features(features)
        model = self._get_model()

        proba = np.asarray(model.predict_proba(X_df), dtype=float)
        predictions = np.asarray(model.predict(X_df), dtype=int)
        risk_levels = model.predict_risk(X_df)

        results: List[RiskPredictionResult] = []
        for i in range(len(X_df)):
            label_id = int(predictions[i])
            label_name = self._label_name(model, label_id)
            probability = float(proba[i, label_id])
            results.append(
                RiskPredictionResult(
                    risk_label=label_name,
                    risk_probability=probability,
                    risk_level=str(risk_levels[i]),
                    model_version=self.model_version or "v1",
                )
            )
        return results, proba

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """校验并规整特征输入。

        Args:
            df: 特征 DataFrame。

        Returns:
            按 FEATURE_VECTOR_COLUMNS 顺序重排的纯特征 DataFrame。

        Raises:
            ValueError: 输入异常。
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("特征输入必须为 pandas DataFrame")
        if df is None or df.empty:
            raise ValueError("特征输入为空")
        if "label" in df.columns:
            raise ValueError("特征输入不应包含 label 列")
        missing = [col for col in FEATURE_VECTOR_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"特征缺少必需列: {missing}")

        X_df = df[FEATURE_VECTOR_COLUMNS]
        if not all(pd.api.types.is_numeric_dtype(X_df[col]) for col in FEATURE_VECTOR_COLUMNS):
            raise ValueError("特征包含非数值列")
        return X_df.copy()

    def _get_model(self) -> Any:
        """通过模型管理获取模型。

        Returns:
            已加载的 XGBoost 模型实例。

        Raises:
            RuntimeError: 模型不存在或未加载。
        """
        model = self.registry.get_model(
            self.model_name,
            version=self.model_version,
            model_factory=self.model_factory,
        )
        if model is None:
            raise RuntimeError(
                f"模型获取失败: {self.model_name}@"
                f"{self.model_version or '(default)'}，模型不存在或未加载"
            )
        if not getattr(model, "is_trained", False):
            raise RuntimeError("模型尚未训练/加载，无法预测")
        self._model = model
        return model

    @staticmethod
    def _label_name(model: Any, label_id: int) -> str:
        """将类别 id 转换为标签名。

        Args:
            model: 模型实例。
            label_id: 类别 id。

        Returns:
            标签名称；未知 id 返回字符串。
        """
        mapping = getattr(model, "label_mapping", None)
        if mapping and label_id in mapping:
            return str(mapping[label_id])
        return str(label_id)
