"""tests/model/test_lstm.py —— LSTM 时序预测模型测试。

逻辑部分（参数、输入校验、元数据保存、LSTMResult）不依赖 torch；
训练/推理/保存加载集成测试在 torch 安装时运行，否则跳过。
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from src.core.models import LSTMResult
from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.lstm import LSTMModel

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


class LSTMModelLogicTest(unittest.TestCase):
    """不依赖 torch 的 LSTM 逻辑测试。"""

    def test_default_params(self):
        """默认参数应符合 LSTM.md 推荐值。"""
        model = LSTMModel()
        self.assertEqual(model.seq_len, 60)
        self.assertEqual(model.hidden_size, 64)
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(model.num_classes, 4)
        self.assertEqual(model.input_size, len(FEATURE_VECTOR_COLUMNS))

    def test_validate_sequence_wrong_dim_raises(self):
        """维度不合法的序列应抛 ValueError。"""
        model = LSTMModel()
        with self.assertRaises(ValueError):
            model._validate_sequence(np.zeros((10,)))
        with self.assertRaises(ValueError):
            model._validate_sequence(np.zeros((10, 5)))

    def test_validate_sequence_correct_shape(self):
        """正确形状的序列应被接受。"""
        model = LSTMModel()
        arr = model._validate_sequence(np.zeros((60, len(FEATURE_VECTOR_COLUMNS))))
        self.assertEqual(arr.shape, (60, len(FEATURE_VECTOR_COLUMNS)))

    def test_predict_not_trained_raises(self):
        """未训练模型预测应抛 RuntimeError。"""
        model = LSTMModel()
        with self.assertRaises(RuntimeError):
            model.predict(np.zeros((60, len(FEATURE_VECTOR_COLUMNS))))

    def test_save_writes_metadata(self):
        """保存应写入特征名、标签映射与训练配置（无需 torch）。"""
        model = LSTMModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = model.save(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "feature_names.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "label_mapping.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "training_config.yaml")))
            self.assertEqual(len(paths), 3)  # net 为 None 时不写 .pth

    def test_lstm_result_to_dict(self):
        """LSTMResult 应正确序列化。"""
        result = LSTMResult(prediction="high_risk", probability=0.9, trend="deteriorating")
        payload = result.to_dict()
        self.assertEqual(payload["prediction"], "high_risk")
        self.assertAlmostEqual(payload["probability"], 0.9)
        self.assertEqual(payload["trend"], "deteriorating")
        self.assertIn("prediction_time", payload)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch 未安装，跳过 LSTM 训练/推理集成测试")
class LSTMModelTorchTest(unittest.TestCase):
    """依赖 torch 的 LSTM 训练/推理/保存加载集成测试。"""

    def test_fit_predict_save_load_roundtrip(self):
        """训练、预测、保存、加载全流程应一致。"""
        model = LSTMModel(epochs=1, batch_size=8)
        n = 16
        X = np.random.randn(n, model.seq_len, model.input_size).astype("float32")
        y = np.random.randint(0, 4, n)

        model.fit(X, y)
        self.assertTrue(model.is_trained)

        result = model.predict(X[0])
        self.assertIsInstance(result, LSTMResult)
        self.assertIn(result.prediction, {"normal", "attention_down", "fatigue", "high_risk"})
        self.assertGreaterEqual(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "lstm_model.pth")))

            loaded = LSTMModel()
            loaded.load(tmpdir)
            self.assertTrue(loaded.is_trained)
            loaded_result = loaded.predict(X[0])
            self.assertEqual(loaded_result.prediction, result.prediction)

    def test_predict_proba_shape(self):
        """预测概率形状应为 (1, num_classes)。"""
        model = LSTMModel(epochs=1, batch_size=8)
        n = 8
        X = np.random.randn(n, model.seq_len, model.input_size).astype("float32")
        y = np.random.randint(0, 4, n)
        model.fit(X, y)
        proba = model.predict_proba(X[0])
        self.assertEqual(proba.shape, (1, model.num_classes))


if __name__ == "__main__":
    unittest.main()
