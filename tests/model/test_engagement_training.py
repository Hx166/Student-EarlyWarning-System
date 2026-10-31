"""阶段 E 基线与指标函数测试，以及 XGBoost 训练冒烟测试。"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from scripts.train_engagement import (
    classification_metrics,
    majority_baseline,
    train_ablation,
)
from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.labels import ENGAGEMENT_LABEL_MAPPING
from src.models.xgboost import XGBoostModel


def test_majority_baseline_predicts_level_2() -> None:
    """多数类基线应全部预测类别 2。"""
    report = majority_baseline([0, 1, 2, 2])
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["per_class"][2]["recall"] == pytest.approx(1.0)
    assert report["per_class"][0]["recall"] == pytest.approx(0.0)
    assert report["roc_auc_ovr"] is None


def test_classification_metrics_with_synthetic_arrays() -> None:
    """合成预测应正确计算 Accuracy、宏 F1、混淆矩阵和 Level 0。"""
    truth = np.array([0, 0, 1, 2, 2, 3])
    prediction = np.array([0, 1, 1, 2, 3, 3])
    report = classification_metrics(truth, prediction)
    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["macro_f1"] == pytest.approx(2 / 3)
    assert report["per_class"][0]["support"] == 2
    assert report["per_class"][0]["recall"] == pytest.approx(0.5)
    assert report["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]


def test_xgboost_model_smoke_400_rows() -> None:
    """400 行合成数据应在 10 秒内完成训练和预测。"""
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(400, len(FEATURE_VECTOR_COLUMNS))),
        columns=FEATURE_VECTOR_COLUMNS,
    )
    y = pd.Series(np.tile(np.arange(4), 100))
    model = XGBoostModel(
        params={
            "n_estimators": 8,
            "max_depth": 2,
            "n_jobs": 1,
            "random_state": 42,
        },
        label_mapping=ENGAGEMENT_LABEL_MAPPING,
    )
    started = time.perf_counter()
    model.fit(X, y)
    predictions = model.predict(X.iloc[:12])
    elapsed = time.perf_counter() - started
    assert model.is_trained
    assert len(predictions) == 12
    assert elapsed < 10


def test_ablation_uses_nonempty_validation_split() -> None:
    """消融函数必须使用传入的 Validation，不能返回空集指标。"""
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(size=(180, len(FEATURE_VECTOR_COLUMNS))),
        columns=FEATURE_VECTOR_COLUMNS,
    )
    frame["engagement"] = np.tile(np.arange(4), 45)
    train = frame.iloc[:120].copy()
    validation = frame.iloc[120:150].copy()
    test = frame.iloc[150:].copy()
    params = {"n_estimators": 6, "max_depth": 2, "n_jobs": 1, "random_state": 42}
    train_metrics, test_metrics, columns = train_ablation(
        train, validation, test, params
    )
    assert len(columns) == 9
    assert sum(item["support"] for item in train_metrics["per_class"]) == 30
    assert sum(item["support"] for item in test_metrics["per_class"]) == 30
