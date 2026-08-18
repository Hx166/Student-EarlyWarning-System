"""tests/service/test_prediction.py —— 实时预测接线模块测试。

覆盖无模型降级、数据不足与正常预测路径。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.registry import ModelRegistry
from src.service.prediction import PredictionService


class StubRiskModel:
    """模拟 XGBoost 预测接口的桩模型。"""

    is_trained: bool = True
    label_mapping: dict = {
        0: "normal",
        1: "attention_down",
        2: "fatigue",
        3: "high_risk",
    }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """返回确定性标签。"""
        return np.array([2] * len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回确定性概率矩阵。"""
        n = len(X)
        proba = np.zeros((n, 4))
        proba[:, 0] = 0.05
        proba[:, 1] = 0.05
        proba[:, 2] = 0.85
        proba[:, 3] = 0.05
        return proba

    def predict_risk(self, X: pd.DataFrame) -> list:
        """返回风险等级。"""
        return ["warning"] * len(X)


def _make_state_record(i: int = 0) -> dict:
    """构造单条 StateRecord 字段字典。"""
    return {
        "record_time": f"2026-08-09T09:00:{i:02d}",
        "avg_ear": 0.30,
        "avg_mar": 0.10,
        "avg_pitch": 5.0,
        "avg_yaw": -2.0,
        "avg_roll": 1.0,
        "perclos": 0.12,
        "attention_score": 80.0,
        "fatigue_score": 20.0,
    }


class PredictionServiceTest(unittest.TestCase):
    """实时预测接线服务测试。"""

    def test_no_model_degrades_gracefully(self):
        """无模型时预测应返回 None 且不抛异常。"""
        svc = PredictionService(model_name="not_exist", min_records=1)
        result = svc.predict(_make_state_record())
        self.assertIsNone(result)

    def test_not_enough_records_returns_none(self):
        """数据不足时不触发预测。"""
        registry = ModelRegistry()
        registry.register("xgboost", StubRiskModel())
        svc = PredictionService(registry=registry, min_records=30)
        result = svc.predict(_make_state_record())
        self.assertIsNone(result)

    def test_predict_with_model(self):
        """累积到足够记录后应返回预测结果。"""
        registry = ModelRegistry()
        registry.register("xgboost", StubRiskModel())
        svc = PredictionService(registry=registry, min_records=2)
        for i in range(2):
            result = svc.predict(_make_state_record(i))
        self.assertIsNotNone(result)
        self.assertEqual(result.risk_label, "fatigue")
        self.assertEqual(result.risk_level, "warning")


if __name__ == "__main__":
    unittest.main()
