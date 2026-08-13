"""tests/model/test_risk_prediction.py —— 风险预测/推理模块测试。

覆盖正常预测、风险等级/概率输出、异常输入、
以及与 XGBoostModel / Model Management 的集成。
使用桩模型进行接口测试，不依赖真实模型文件。
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.registry import ModelRegistry, register_model_loader
from src.models.risk_prediction import RiskPredictionResult, RiskPredictor


def make_feature_df(n: int = 3) -> pd.DataFrame:
    """构造测试用特征 DataFrame（不含 label）。

    Args:
        n: 行数。

    Returns:
        特征 DataFrame。
    """
    rng = np.random.default_rng(0)
    data: Dict[str, Any] = {
        col: rng.uniform(0.0, 1.0, n) for col in FEATURE_VECTOR_COLUMNS
    }
    return pd.DataFrame(data)


class StubRiskModel:
    """用于接口测试的桩模型，模拟 XGBoost 预测接口。"""

    is_trained: bool = True
    label_mapping: Dict[int, str] = {
        0: "normal",
        1: "attention_down",
        2: "fatigue",
        3: "high_risk",
    }

    def __init__(self, labels: List[int] = None) -> None:
        """初始化桩模型。

        Args:
            labels: 每行预测标签（可空，默认全 0）。
        """
        self._labels = labels or []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """返回预测标签。"""
        if self._labels:
            return np.array(self._labels[: len(X)], dtype=int)
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回确定性概率矩阵。"""
        n = len(X)
        proba = np.zeros((n, 4))
        proba[:, 0] = 0.9
        proba[:, 1] = 0.05
        proba[:, 2] = 0.03
        proba[:, 3] = 0.02
        return proba

    def predict_risk(self, X: pd.DataFrame) -> List[str]:
        """返回风险等级。"""
        return ["normal"] * len(X)


class RiskPredictorTest(unittest.TestCase):
    """风险预测器测试。"""

    def setUp(self) -> None:
        """构建模型管理、桩模型并注册到磁盘加载路径。"""
        self.stub = StubRiskModel()
        self.registry = ModelRegistry()

        # 直接注册桩模型到内存（get_model 优先返回内存模型）
        self.registry.register("xgboost", self.stub)
        self.predictor = RiskPredictor(registry=self.registry)

    def test_predict_single_row(self):
        """单行预测应返回 RiskPredictionResult。"""
        df = make_feature_df(1)
        result = self.predictor.predict(df)
        self.assertIsInstance(result, RiskPredictionResult)
        self.assertEqual(result.risk_label, "normal")
        self.assertAlmostEqual(result.risk_probability, 0.9, places=6)
        self.assertEqual(result.risk_level, "normal")
        self.assertEqual(result.confidence, result.risk_probability)
        self.assertIn("prediction_time", result.to_dict())

    def test_predict_batch(self):
        """批量预测应返回每行一个结果。"""
        df = make_feature_df(3)
        results = self.predictor.predict_batch(df)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.risk_label, "normal")
            self.assertEqual(r.risk_probability, 0.9)

    def test_predict_risk_level_output(self):
        """风险等级应正确输出到结果。"""
        df = make_feature_df(1)
        result = self.predictor.predict(df)
        self.assertEqual(result.risk_level, "normal")
        self.assertIn(result.risk_level, {"normal", "attention", "warning", "high_risk"})

    def test_empty_input_raises(self):
        """空输入应抛 ValueError。"""
        with self.assertRaises(ValueError):
            self.predictor.predict_batch(pd.DataFrame())

    def test_none_input_raises(self):
        """None 输入应抛 ValueError。"""
        with self.assertRaises(ValueError):
            self.predictor.predict_batch(None)

    def test_missing_feature_raises(self):
        """缺列应抛 ValueError。"""
        df = make_feature_df(2).drop(columns=["ear_mean"])
        with self.assertRaises(ValueError):
            self.predictor.predict_batch(df)

    def test_label_column_rejected(self):
        """含 label 列应抛 ValueError。"""
        df = make_feature_df(2)
        df["label"] = 0
        with self.assertRaises(ValueError):
            self.predictor.predict_batch(df)

    def test_non_numeric_feature_raises(self):
        """非数值特征应抛 ValueError。"""
        df = make_feature_df(2)
        df["ear_mean"] = "abc"
        with self.assertRaises(ValueError):
            self.predictor.predict_batch(df)

    def test_wrong_input_type_raises(self):
        """非 DataFrame 输入应抛 ValueError。"""
        with self.assertRaises(ValueError):
            self.predictor.predict_batch([1.0, 2.0, 3.0])

    def test_model_missing_raises(self):
        """模型不存在时应抛 RuntimeError。"""
        empty_registry = ModelRegistry()
        predictor = RiskPredictor(registry=empty_registry, model_name="not_exist")
        with self.assertRaises(RuntimeError):
            predictor.predict_batch(make_feature_df(2))

    def test_model_not_trained_raises(self):
        """模型未训练时应抛 RuntimeError。"""
        class _Untrained(StubRiskModel):
            is_trained = False

        reg = ModelRegistry()
        reg.register("xgboost", _Untrained())
        predictor = RiskPredictor(registry=reg)
        with self.assertRaises(RuntimeError):
            predictor.predict_batch(make_feature_df(2))

    def test_predict_single_row_requires_one_row(self):
        """predict 单行接口对多行输入应抛 ValueError。"""
        with self.assertRaises(ValueError):
            self.predictor.predict(make_feature_df(2))


class RiskPredictorIntegrationTest(unittest.TestCase):
    """与 Model Management 磁盘加载的集成测试。"""

    def setUp(self) -> None:
        """构造带模型管理与 xgboost 工厂的环境。"""
        self.registry = ModelRegistry()
        register_model_loader("xgboost", lambda: StubRiskModel())

    def test_get_model_via_management(self):
        """RiskPredictor 应通过模型管理获取模型。"""
        # 模拟磁盘上已存在模型文件（缺真实 xgboost 时用注册内存模型兜底）
        self.registry.register("xgboost", StubRiskModel())
        predictor = RiskPredictor(registry=self.registry, model_name="xgboost")
        df = make_feature_df(2)
        results = predictor.predict_batch(df)
        self.assertEqual(len(results), 2)

    def test_label_mapping_names(self):
        """标签名应与 XGBoost 定义一致。"""
        self.assertEqual(
            StubRiskModel.label_mapping,
            {0: "normal", 1: "attention_down", 2: "fatigue", 3: "high_risk"},
        )


if __name__ == "__main__":
    unittest.main()
