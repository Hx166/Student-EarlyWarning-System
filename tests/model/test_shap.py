"""tests/model/test_shap.py —— SHAP 可解释性模块测试。

使用桩解释器与桩模型进行接口测试，不依赖真实 shap / xgboost。
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.shap import ShapExplainer


class FakeExplainer:
    """模拟 shap.TreeExplainer 接口的桩解释器。"""

    def __init__(self) -> None:
        """初始化期望值。"""
        self.expected_value = [0.1, 0.2, 0.3, 0.4]

    def shap_values(self, X: pd.DataFrame) -> List[np.ndarray]:
        """返回确定性多分类 SHAP 值。"""
        n = len(X)
        f = len(FEATURE_VECTOR_COLUMNS)
        base = np.arange(f).reshape(1, -1) * 0.01
        return [np.full((n, f), i * 0.1) + base for i in range(4)]


class StubModel:
    """模拟 XGBoostModel 的桩模型。"""

    is_trained: bool = True
    feature_names: List[str] = list(FEATURE_VECTOR_COLUMNS)
    label_mapping: Dict[int, str] = {
        0: "normal",
        1: "attention_down",
        2: "fatigue",
        3: "high_risk",
    }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """返回确定性标签。"""
        return np.array([2] * len(X), dtype=int)


class ShapExplainerTest(unittest.TestCase):
    """SHAP 解释器测试。"""

    def _features(self, n: int = 3) -> pd.DataFrame:
        """构造测试特征 DataFrame。"""
        return pd.DataFrame({col: [0.5] * n for col in FEATURE_VECTOR_COLUMNS})

    def test_not_trained_raises(self):
        """未训练模型应抛 ValueError。"""
        with self.assertRaises(ValueError):
            ShapExplainer(object())

    def test_global_importance_structure(self):
        """全局重要性应返回降序排列的特征贡献列表。"""
        explainer = ShapExplainer(StubModel(), explainer=FakeExplainer())
        ranking = explainer.global_importance(self._features())
        self.assertEqual(len(ranking), len(FEATURE_VECTOR_COLUMNS))
        self.assertIn("feature", ranking[0])
        self.assertIn("importance", ranking[0])
        imps = [item["importance"] for item in ranking]
        self.assertEqual(imps, sorted(imps, reverse=True))

    def test_global_importance_specific_class(self):
        """指定类别应返回该类别的重要性排序。"""
        explainer = ShapExplainer(StubModel(), explainer=FakeExplainer())
        ranking = explainer.global_importance(self._features(), class_index=2)
        self.assertEqual(len(ranking), len(FEATURE_VECTOR_COLUMNS))

    def test_local_explanation_structure(self):
        """局部解释应返回类别、基准值与按贡献排序的特征列表。"""
        explainer = ShapExplainer(StubModel(), explainer=FakeExplainer())
        result = explainer.local_explanation(self._features(n=1))
        self.assertEqual(result["class_id"], 2)
        self.assertEqual(result["class_name"], "fatigue")
        self.assertIn("base_value", result)
        self.assertEqual(len(result["contributions"]), len(FEATURE_VECTOR_COLUMNS))
        abs_vals = [abs(c["contribution"]) for c in result["contributions"]]
        self.assertEqual(abs_vals, sorted(abs_vals, reverse=True))


if __name__ == "__main__":
    unittest.main()
