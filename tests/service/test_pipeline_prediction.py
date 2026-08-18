"""tests/service/test_pipeline_prediction.py —— Pipeline 预测接入降级测试。"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.models.registry import ModelRegistry
from src.service.pipeline import PipelineService
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

    def __init__(self, label: int = 3, proba: float = 0.9, level: str = "high_risk"):
        """初始化桩模型。

        Args:
            label: 预测标签 id。
            proba: 预测概率。
            level: 风险等级。
        """
        self._label = label
        self._proba = proba
        self._level = level

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """返回确定性标签。"""
        return np.array([self._label] * len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回确定性概率矩阵。"""
        n = len(X)
        proba = np.zeros((n, 4))
        proba[:, self._label] = self._proba
        return proba

    def predict_risk(self, X: pd.DataFrame) -> list:
        """返回风险等级。"""
        return [self._level] * len(X)


class _FakeDb:
    """记录调用的简化数据库桩。"""

    def __init__(self):
        self.predictions = []
        self.warnings = []

    def insert_prediction(self, **kwargs):
        self.predictions.append(kwargs)
        return len(self.predictions)

    def insert_warning(self, **kwargs):
        self.warnings.append(kwargs)
        return len(self.warnings)


def _state_record() -> dict:
    """构造单条 StateRecord 字段字典。"""
    return {
        "record_time": "2026-08-09T09:00:00",
        "avg_ear": 0.30,
        "avg_mar": 0.10,
        "avg_pitch": 5.0,
        "avg_yaw": -2.0,
        "avg_roll": 1.0,
        "perclos": 0.12,
        "attention_score": 40.0,
        "fatigue_score": 60.0,
    }


class PipelinePredictionTest(unittest.TestCase):
    """Pipeline 预测接入测试。"""

    def setUp(self):
        """构建带预测服务的 pipeline 与桩数据库。"""
        self.camera = mock.MagicMock()
        self.cv = mock.MagicMock()
        self.db = _FakeDb()
        self.pipeline = PipelineService(
            camera=self.camera, cv=self.cv, db=self.db,
            student_id=1, session_id=2,
        )

    def test_no_model_degrades_gracefully(self):
        """无模型时预测应静默跳过，不写入 PredictionLog。"""
        self.pipeline.prediction_service = PredictionService(
            model_name="not_exist", min_records=1
        )
        self.pipeline._maybe_predict(_state_record())
        self.assertEqual(len(self.db.predictions), 0)
        self.assertEqual(len(self.db.warnings), 0)

    def test_high_risk_triggers_warning(self):
        """高风险预测写入 PredictionLog 并触发 WarningLog。"""
        registry = ModelRegistry()
        registry.register("xgboost", StubRiskModel())
        self.pipeline.prediction_service = PredictionService(
            registry=registry, min_records=1
        )
        self.pipeline._maybe_predict(_state_record())
        self.assertEqual(len(self.db.predictions), 1)
        self.assertEqual(len(self.db.warnings), 1)
        self.assertEqual(self.db.predictions[0]["risk_level"], "high_risk")
        self.assertEqual(self.db.warnings[0]["warning_type"], "model_prediction")
        self.assertEqual(self.db.warnings[0]["warning_level"], "high")


if __name__ == "__main__":
    unittest.main()
