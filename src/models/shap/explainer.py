"""src.models.shap.explainer —— SHAP 可解释性分析。

基于已训练的 XGBoost 模型，使用 shap.TreeExplainer 生成特征贡献，
提供全局解释（特征重要性排序）与局部解释（单样本特征贡献）。

输出均为可序列化的 dict/list 结构，供 API / Dashboard 直接使用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShapExplainer:
    """XGBoost 模型 SHAP 可解释性分析器。

    Attributes:
        feature_names: 特征列名列表。
        _model: 已训练的 XGBoost 模型实例。
        _explainer: shap.TreeExplainer 实例（延迟创建）。
    """

    def __init__(self, model: Any, explainer: Optional[Any] = None) -> None:
        """初始化 SHAP 解释器。

        Args:
            model: 已训练的 XGBoost 模型实例（需具备 is_trained）。
            explainer: 可选的已构造 shap 解释器（测试注入用）。

        Raises:
            ValueError: 模型未训练时抛出。
        """
        if not getattr(model, "is_trained", False):
            raise ValueError("模型尚未训练/加载，无法解释")
        self._model = model
        self._explainer = explainer
        self.feature_names: List[str] = list(
            getattr(model, "feature_names", None) or FEATURE_VECTOR_COLUMNS
        )
        logger.info("ShapExplainer 已初始化，特征数=%s", len(self.feature_names))

    def global_importance(
        self, X: pd.DataFrame, class_index: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """计算全局特征重要性排序。

        Args:
            X: 特征 DataFrame。
            class_index: 目标类别 id；为空时对所有类别求平均。

        Returns:
            按重要性降序排列的特征贡献列表，
            元素形如 {"feature": str, "importance": float}。
        """
        explainer = self._ensure_explainer()
        raw = explainer.shap_values(X)
        values = self._to_sample_matrix(raw, class_index)
        importance = np.abs(values).mean(axis=0)
        ranking = [
            {"feature": self.feature_names[i], "importance": float(importance[i])}
            for i in range(len(self.feature_names))
        ]
        ranking.sort(key=lambda item: item["importance"], reverse=True)
        return ranking

    def local_explanation(self, features: pd.DataFrame) -> Dict[str, Any]:
        """计算单样本（单行）局部特征贡献。

        Args:
            features: 单行特征 DataFrame。

        Returns:
            局部解释字典，包含预测类别、基准值与各特征贡献排序。
        """
        explainer = self._ensure_explainer()
        class_id = int(self._model.predict(features)[0])
        raw = explainer.shap_values(features)
        values = self._to_sample_matrix(raw, class_id)
        row = np.asarray(values[0], dtype=float)

        contributions = [
            {"feature": self.feature_names[i], "contribution": float(row[i])}
            for i in range(len(self.feature_names))
        ]
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)

        class_name = self._class_name(class_id)
        return {
            "class_id": class_id,
            "class_name": class_name,
            "base_value": float(self._base_value(explainer, class_id)),
            "contributions": contributions,
        }

    def _ensure_explainer(self) -> Any:
        """延迟创建 shap.TreeExplainer。

        Returns:
            shap 解释器实例。

        Raises:
            ImportError: shap 未安装时抛出。
        """
        if self._explainer is None:
            try:
                import shap
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "未安装 shap，无法进行可解释性分析，请执行 pip install shap"
                ) from exc
            self._explainer = shap.TreeExplainer(self._model.model)
        return self._explainer

    def _to_sample_matrix(
        self, raw: Any, class_index: Optional[int]
    ) -> np.ndarray:
        """将 shap 原始输出规整为 (n_samples, n_features) 矩阵。

        Args:
            raw: explainer.shap_values 的原始返回。
            class_index: 目标类别 id；多分类且为空时对所有类别求平均。

        Returns:
            形状为 (n_samples, n_features) 的 ndarray。
        """
        if isinstance(raw, list):
            if class_index is not None:
                return np.asarray(raw[class_index], dtype=float)
            stacked = np.stack([np.asarray(v, dtype=float) for v in raw], axis=0)
            return np.abs(stacked).mean(axis=0)
        return np.asarray(raw, dtype=float)

    def _base_value(self, explainer: Any, class_index: int) -> float:
        """提取指定类别的基准值（期望预测值）。

        Args:
            explainer: shap 解释器。
            class_index: 类别 id。

        Returns:
            基准值。
        """
        expected = getattr(explainer, "expected_value", 0.0)
        arr = np.asarray(expected, dtype=float)
        if arr.ndim > 0 and class_index < len(arr):
            return float(arr[class_index])
        return float(arr)

    def _class_name(self, class_id: int) -> str:
        """将类别 id 转换为名称。

        Args:
            class_id: 类别 id。

        Returns:
            类别名称；未知 id 返回字符串。
        """
        mapping = getattr(self._model, "label_mapping", None)
        if mapping and class_id in mapping:
            return str(mapping[class_id])
        return str(class_id)
