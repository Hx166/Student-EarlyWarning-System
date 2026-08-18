"""tests/api/test_predict.py —— 预测相关 API 测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.api.app import create_app
from src.api.service import prediction as pred_svc
from src.database import DatabaseService
from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.registry import ModelRegistry


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


def _feature_dict() -> dict:
    """构造完整 11 列特征向量字典。"""
    return {col: 0.0 for col in FEATURE_VECTOR_COLUMNS}


class PredictApiTest(unittest.TestCase):
    """预测 API 测试。"""

    def setUp(self):
        """重置单例并注入桩模型。"""
        pred_svc.reset()
        registry: ModelRegistry = pred_svc.get_model_registry()
        registry.register("xgboost", StubRiskModel())

        self.db = DatabaseService(db_path=":memory:").connect()
        self.app = create_app(database=self.db)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        sid = self.db.upsert_student("S0001", "张三")
        seid = self.db.create_session("高数", "李老师", "A101")
        self.db.insert_prediction(
            sid, seid, "fatigue", 0.72, "warning", "v1",
        )

    def tearDown(self):
        """关闭数据库并重置单例。"""
        self.db.close()
        pred_svc.reset()

    def test_predict_normal(self):
        """正常特征向量应返回预测结果。"""
        res = self.client.post("/api/predict", json={"features": _feature_dict()})
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["code"], 0)
        self.assertEqual(payload["data"]["risk_label"], "fatigue")
        self.assertEqual(payload["data"]["risk_level"], "warning")
        self.assertAlmostEqual(payload["data"]["risk_probability"], 0.85, places=6)

    def test_predict_missing_column(self):
        """缺列应返回 400。"""
        features = _feature_dict()
        del features["ear_mean"]
        res = self.client.post("/api/predict", json={"features": features})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["code"], 1)

    def test_predict_with_label_column(self):
        """含 label 列应返回 400。"""
        features = _feature_dict()
        features["label"] = 0
        res = self.client.post("/api/predict", json={"features": features})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["code"], 1)

    def test_predict_missing_features_field(self):
        """缺少 features 字段应返回 400。"""
        res = self.client.post("/api/predict", json={})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["code"], 1)

    def test_predictions(self):
        """预测记录接口应返回写入的记录。"""
        res = self.client.get("/api/predictions")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["code"], 0)
        self.assertEqual(len(payload["data"]), 1)
        self.assertEqual(payload["data"][0]["risk_label"], "fatigue")

    def test_model_info(self):
        """模型信息接口应返回模型状态。"""
        res = self.client.get("/api/model/info")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["code"], 0)
        self.assertEqual(payload["data"]["name"], "xgboost")
        self.assertEqual(payload["data"]["version"], "v1")
        self.assertTrue(payload["data"]["loaded"])


if __name__ == "__main__":
    unittest.main()
