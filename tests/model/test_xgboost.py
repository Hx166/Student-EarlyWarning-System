"""tests/model/test_xgboost.py —— XGBoost 模型测试。

覆盖训练、预测、风险等级映射、评估、保存/加载与输入校验。
xgboost/sklearn 缺失时，训练与评估用例跳过（沿用项目既有模式）；
接口级用例使用桩模型验证，不依赖 xgboost 也可运行。
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.xgboost import (
    DEFAULT_PARAMS,
    LABEL_MAPPING,
    LABEL_NAMES,
    RISK_THRESHOLDS,
    XGBoostModel,
)

XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
FULL_AVAILABLE = XGBOOST_AVAILABLE and SKLEARN_AVAILABLE


def make_feature_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """生成用于测试的特征 DataFrame（含 label）。

    Args:
        n: 样本数。
        seed: 随机种子。

    Returns:
        特征 DataFrame。
    """
    rng = np.random.default_rng(seed)
    data: Dict[str, Any] = {
        "ear_mean": rng.uniform(0.2, 0.35, n),
        "ear_std": rng.uniform(0.01, 0.05, n),
        "blink_rate": rng.uniform(5, 40, n),
        "perclos": rng.uniform(0.05, 0.5, n),
        "pitch_mean": rng.uniform(-10, 40, n),
        "yaw_mean": rng.uniform(-20, 20, n),
        "roll_mean": rng.uniform(-10, 10, n),
        "fatigue_score": rng.uniform(0, 100, n),
        "attention_score": rng.uniform(0, 100, n),
        "head_down_duration": rng.uniform(0, 300, n),
        "face_lost_duration": rng.uniform(0, 120, n),
    }
    df = pd.DataFrame(data)
    # 简化的规则标签，保证 4 类均出现
    fatigue = df["fatigue_score"].values
    attention = df["attention_score"].values
    labels = np.full(n, 0, dtype=int)
    labels[(attention < 60) & (fatigue < 60)] = 1
    labels[(fatigue >= 60) & (attention >= 50)] = 2
    labels[(fatigue >= 60) & (attention < 50)] = 3
    df["label"] = labels
    return df


class FakeModel:
    """用于接口测试的桩模型，模拟 xgboost 分类器。"""

    def __init__(self, n_classes: int = 4) -> None:
        """初始化桩模型。

        Args:
            n_classes: 类别数。
        """
        self.n_classes = n_classes

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """返回固定标签。"""
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回确定性概率矩阵。"""
        n = len(X)
        proba = np.full((n, self.n_classes), 0.25)
        proba[:, 0] = 0.97
        proba[:, 1:] = 0.01
        return proba

    def save_model(self, path: str) -> None:
        """写入模型文件桩。"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("fake-model")

    def load_model(self, path: str) -> None:
        """读取模型文件桩。"""
        with open(path, "r", encoding="utf-8") as fh:
            fh.read()


def make_trained_model() -> XGBoostModel:
    """构造已训练（桩）的 XGBoostModel 实例。

    Returns:
        带桩模型且 is_trained=True 的实例。
    """
    model_wrapper = XGBoostModel()
    model_wrapper.model = FakeModel()
    model_wrapper.feature_names = list(FEATURE_VECTOR_COLUMNS)
    model_wrapper.is_trained = True
    return model_wrapper


class XGBoostConstantsTest(unittest.TestCase):
    """常量定义测试。"""

    def test_label_mapping_has_four_classes(self):
        """标签映射应包含 0~3 四类。"""
        self.assertEqual(sorted(LABEL_MAPPING.keys()), [0, 1, 2, 3])
        self.assertEqual(len(LABEL_NAMES), 4)

    def test_default_params_match_document(self):
        """默认参数应包含文档规定的核心参数。"""
        self.assertEqual(DEFAULT_PARAMS["objective"], "multi:softprob")
        self.assertEqual(DEFAULT_PARAMS["num_class"], 4)
        self.assertEqual(DEFAULT_PARAMS["max_depth"], 6)
        self.assertEqual(DEFAULT_PARAMS["learning_rate"], 0.1)
        self.assertEqual(DEFAULT_PARAMS["random_state"], 42)

    def test_risk_thresholds_order(self):
        """风险阈值应按 0.30/0.60/0.80 升序。"""
        thresholds = [t for t, _ in RISK_THRESHOLDS]
        self.assertEqual(thresholds, [0.30, 0.60, 0.80])


class XGBoostInterfaceTest(unittest.TestCase):
    """接口级测试（不依赖 xgboost/sklearn）。"""

    def test_predict_without_training_raises(self):
        """未训练时 predict 应抛出 RuntimeError。"""
        model = XGBoostModel()
        df = make_feature_df(5).drop(columns=["label"])
        with self.assertRaises(RuntimeError):
            model.predict(df)

    def test_predict_with_fake_model(self):
        """预测应返回 4 类标签且形状正确。"""
        model = make_trained_model()
        df = make_feature_df(10).drop(columns=["label"])
        preds = model.predict(df)
        self.assertEqual(preds.shape, (10,))
        self.assertTrue(set(preds).issubset({0, 1, 2, 3}))

    def test_predict_proba_with_fake_model(self):
        """概率预测应返回 (n, 4) 矩阵。"""
        model = make_trained_model()
        df = make_feature_df(8).drop(columns=["label"])
        proba = model.predict_proba(df)
        self.assertEqual(proba.shape, (8, 4))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_risk_mapping(self):
        """风险等级应按概率峰值阈值映射。"""
        model = make_trained_model()
        df = make_feature_df(6).drop(columns=["label"])
        levels = model.predict_risk(df)
        # 桩概率峰值为 0.97 → high_risk
        self.assertEqual(levels, ["high_risk"] * 6)

    def test_predict_risk_threshold_boundary(self):
        """风险阈值边界应正确映射。"""
        model = make_trained_model()

        class _Weights(FakeModel):
            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
                n = len(X)
                peak = np.array([0.29, 0.30, 0.60, 0.80, 0.81][:n])
                proba = np.zeros((n, 4))
                proba[:, 0] = peak
                proba[:, 1:] = (1.0 - peak)[:, None] / 3.0
                return proba

        model.model = _Weights()
        df = make_feature_df(5).drop(columns=["label"])
        levels = model.predict_risk(df)
        self.assertEqual(levels, ["normal", "attention", "warning", "warning", "high_risk"])

    def test_check_features_missing_column(self):
        """缺列时应抛出 ValueError。"""
        model = XGBoostModel()
        df = make_feature_df(5).drop(columns=["label"]).drop(columns=["ear_mean"])
        with self.assertRaises(ValueError):
            model._check_features(df)

    def test_check_features_rejects_label(self):
        """含 label 列时应抛出 ValueError。"""
        model = XGBoostModel()
        df = make_feature_df(5)
        with self.assertRaises(ValueError):
            model._check_features(df)

    def test_fit_missing_label_raises(self):
        """DataFrame 无 label 且未提供 y 时应抛出 ValueError。"""
        model = XGBoostModel()
        df = make_feature_df(5).drop(columns=["label"])
        with self.assertRaises(ValueError):
            model.fit(df)

    @unittest.skipUnless(FULL_AVAILABLE, "需要 xgboost 与 scikit-learn")
    def test_fit_predict_evaluate_save_load(self):
        """真实训练-预测-评估-保存-加载全链路。"""
        df = make_feature_df(120)
        X = df.drop(columns=["label"])
        y = df["label"].values

        model = XGBoostModel(
            params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1}
        )
        model.fit(X, y)
        self.assertTrue(model.is_trained)

        preds = model.predict(X)
        self.assertEqual(preds.shape, (120,))
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (120, 4))

        report = model.evaluate(X, y)
        self.assertIn("accuracy", report)
        self.assertIn("precision", report)
        self.assertIn("recall", report)
        self.assertIn("f1", report)
        self.assertIn("confusion_matrix", report)
        self.assertIn("roc_auc", report)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = model.save(tmpdir)
            self.assertEqual(len(saved), 4)
            for path in saved:
                self.assertTrue(os.path.exists(path))

            loaded = XGBoostModel().load(tmpdir)
            self.assertTrue(loaded.is_trained)
            np.testing.assert_array_equal(loaded.predict(X), preds)

    @unittest.skipUnless(XGBOOST_AVAILABLE, "需要 xgboost")
    def test_fit_from_dataframe_with_label(self):
        """DataFrame 含 label 列时可直接训练。"""
        df = make_feature_df(80)
        model = XGBoostModel(
            params={"n_estimators": 15, "max_depth": 3}
        )
        model.fit(df)
        self.assertTrue(model.is_trained)

    @unittest.skipUnless(FULL_AVAILABLE, "需要 xgboost 与 scikit-learn")
    def test_save_and_load_roundtrip_with_metadata(self):
        """保存的元数据文件应包含 feature_names 与 label_mapping。"""
        df = make_feature_df(60)
        model = XGBoostModel(params={"n_estimators": 10})
        model.fit(df.drop(columns=["label"]), df["label"].values)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            with open(os.path.join(tmpdir, "feature_names.json"), "r", encoding="utf-8") as fh:
                feature_names = json.load(fh)
            self.assertEqual(feature_names, FEATURE_VECTOR_COLUMNS)
            with open(os.path.join(tmpdir, "label_mapping.json"), "r", encoding="utf-8") as fh:
                mapping = json.load(fh)
            self.assertEqual(set(mapping.keys()), {"0", "1", "2", "3"})
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "training_config.yaml")))


if __name__ == "__main__":
    unittest.main()
