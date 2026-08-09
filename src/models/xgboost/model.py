"""src/models/xgboost/model.py —— XGBoost 课堂状态分类模型。

依据 docs/05-机器学习/XGBoost.md 与 Evaluation.md 实现：
- 4 类状态分类（0 正常 / 1 注意力下降 / 2 疲劳 / 3 高风险）；
- 输入为 Feature Engineering 输出的 11 列特征向量；
- 提供训练、预测、风险等级映射、模型保存/加载与评估接口。

说明：本模块只负责模型训练与推理，不负责特征采集、
模型统一注册（ModelRegistry）或预测服务接入。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 4 类状态标签（与 XGBoost.md 第 6 节、FeatureEngineering.md 一致）
LABEL_MAPPING: Dict[int, str] = {
    0: "normal",
    1: "attention_down",
    2: "fatigue",
    3: "high_risk",
}

# 4 类状态分类标签名（按类别 id 排序）
LABEL_NAMES: List[str] = [LABEL_MAPPING[i] for i in range(len(LABEL_MAPPING))]

# 默认模型参数（XGBoost.md 第 9 节建议初始参数）
DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": 4,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "min_child_weight": 1,
    "random_state": 42,
    "eval_metric": "mlogloss",
}

# 风险等级阈值（XGBoost.md 第 13 节）
RISK_THRESHOLDS: List[Tuple[float, str]] = [
    (0.30, "normal"),
    (0.60, "attention"),
    (0.80, "warning"),
]


class XGBoostModel:
    """XGBoost 4 类课堂状态分类模型。

    Attributes:
        params: 模型超参数。
        model: xgboost XGBClassifier 实例（训练后可用）。
        feature_names: 训练时使用的特征列名。
        label_mapping: 类别 id 到语义名称的映射。
        is_trained: 模型是否已完成训练。
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        label_mapping: Optional[Dict[int, str]] = None,
    ) -> None:
        """初始化 XGBoost 模型。

        Args:
            params: 覆盖默认参数的模型超参数字典（可空）。
            label_mapping: 类别 id 到名称的映射（可空，默认使用 LABEL_MAPPING）。
        """
        merged = dict(DEFAULT_PARAMS)
        if params:
            merged.update(params)
        self.params: Dict[str, Any] = merged
        self.label_mapping: Dict[int, str] = dict(
            label_mapping if label_mapping is not None else LABEL_MAPPING
        )
        self.model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        logger.info("XGBoostModel 已初始化")

    @staticmethod
    def _model_class() -> Any:
        """返回 xgboost 的 XGBClassifier 类。

        Returns:
            xgboost.XGBClassifier 类。

        Raises:
            ImportError: xgboost 未安装。
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "未安装 xgboost，无法使用 XGBoost 模型，请执行 pip install xgboost"
            ) from exc
        return XGBClassifier

    @staticmethod
    def _check_features(df: pd.DataFrame) -> pd.DataFrame:
        """校验并规整特征输入。

        Args:
            df: 特征 DataFrame（列需包含 FEATURE_VECTOR_COLUMNS）。

        Returns:
            按标准特征顺序重排后的 DataFrame。

        Raises:
            ValueError: 缺少必需特征列或包含 label 列。
        """
        if df is None or df.empty:
            raise ValueError("特征 DataFrame 为空")
        if "label" in df.columns:
            raise ValueError("特征 DataFrame 不应包含 label 列")
        missing = [col for col in FEATURE_VECTOR_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"特征缺少必需列: {missing}")
        return df[FEATURE_VECTOR_COLUMNS].copy()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Sequence[int]] = None,
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "XGBoostModel":
        """训练模型。

        Args:
            X: 特征 DataFrame（含 FEATURE_VECTOR_COLUMNS 列）或特征矩阵；
                若为 DataFrame 且 y 为 None，则读取 label 列作为标签。
            y: 标签序列（0~3），X 为 DataFrame 且不含 label 列时必须提供。
            sample_weight: 样本权重（可空）。

        Returns:
            训练后的 self。

        Raises:
            ValueError: 标签缺失或特征/标签不一致。
            ImportError: xgboost 未安装。
        """
        if isinstance(X, pd.DataFrame):
            if y is None:
                if "label" not in X.columns:
                    raise ValueError("DataFrame 缺少 label 列且未提供 y")
                labels = X["label"].astype(int).values
                X_df = self._check_features(X.drop(columns=["label"]))
            else:
                X_df = self._check_features(X)
                labels = np.asarray(y, dtype=int)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim != 2 or X_arr.shape[1] != len(FEATURE_VECTOR_COLUMNS):
                raise ValueError(
                    "特征矩阵必须为 (n_samples, %d)" % len(FEATURE_VECTOR_COLUMNS)
                )
            if y is None:
                raise ValueError("特征矩阵输入时必须提供 y")
            X_df = pd.DataFrame(X_arr, columns=FEATURE_VECTOR_COLUMNS)
            labels = np.asarray(y, dtype=int)

        if len(X_df) != len(labels):
            raise ValueError("特征与标签数量不一致")
        invalid = set(np.unique(labels)) - set(self.label_mapping.keys())
        if invalid:
            raise ValueError(f"标签包含非法类别: {sorted(invalid)}")

        model_cls = self._model_class()
        model = model_cls(**self.params, verbosity=0)
        model.fit(X_df, labels, sample_weight=sample_weight)
        self.model = model
        self.feature_names = list(FEATURE_VECTOR_COLUMNS)
        self.is_trained = True
        logger.info("XGBoost 模型训练完成，样本数=%s", len(labels))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测类别标签。

        Args:
            X: 特征 DataFrame 或特征矩阵。

        Returns:
            类别标签数组（0~3）。

        Raises:
            RuntimeError: 模型未训练。
        """
        self._require_trained()
        X_df = self._prepare_predict_input(X)
        return np.asarray(self.model.predict(X_df), dtype=int)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测 4 类概率。

        Args:
            X: 特征 DataFrame 或特征矩阵。

        Returns:
            形状为 (n_samples, 4) 的概率矩阵。

        Raises:
            RuntimeError: 模型未训练。
        """
        self._require_trained()
        X_df = self._prepare_predict_input(X)
        return np.asarray(self.model.predict_proba(X_df), dtype=float)

    def predict_risk(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """由预测概率映射风险等级（XGBoost.md 第 13 节阈值）。

        以概率最高的类别概率作为风险概率，按文档阈值映射风险等级。

        Args:
            X: 特征 DataFrame 或特征矩阵。

        Returns:
            风险等级字符串列表。
        """
        proba = self.predict_proba(X)
        max_proba = proba.max(axis=1)
        levels: List[str] = []
        for p in max_proba:
            if p < RISK_THRESHOLDS[0][0]:
                levels.append("normal")
            elif p < RISK_THRESHOLDS[1][0]:
                levels.append("attention")
            elif p <= RISK_THRESHOLDS[2][0]:
                levels.append("warning")
            else:
                levels.append("high_risk")
        return levels

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Sequence[int],
    ) -> Dict[str, Any]:
        """在测试集上评估模型并返回指标。

        包含 Accuracy / Precision / Recall / F1 / ROC-AUC 与混淆矩阵。

        Args:
            X: 特征 DataFrame 或特征矩阵。
            y: 真实标签序列。

        Returns:
            评价指标字典。

        Raises:
            RuntimeError: 模型未训练。
        """
        self._require_trained()
        y_true = np.asarray(y, dtype=int)
        y_pred = self.predict(X)
        proba = self.predict_proba(X)
        return self._sklearn_metrics(y_true, y_pred, proba)

    def save(self, model_dir: str) -> List[str]:
        """保存模型与元数据。

        保存内容（按 XGBoost.md 第 11 节）：
        - xgboost_model.json：模型文件；
        - feature_names.json：特征列名；
        - label_mapping.json：类别映射；
        - training_config.yaml：训练参数。

        Args:
            model_dir: 保存目录。

        Returns:
            实际写入的文件路径列表。

        Raises:
            RuntimeError: 模型未训练。
            ImportError: xgboost 未安装。
        """
        self._require_trained()
        os.makedirs(model_dir, exist_ok=True)
        paths: List[str] = []

        model_path = os.path.join(model_dir, "xgboost_model.json")
        self.model.save_model(model_path)
        paths.append(model_path)

        feature_path = os.path.join(model_dir, "feature_names.json")
        with open(feature_path, "w", encoding="utf-8") as fh:
            json.dump(self.feature_names, fh, ensure_ascii=False, indent=2)
        paths.append(feature_path)

        label_path = os.path.join(model_dir, "label_mapping.json")
        mapping = {str(k): v for k, v in self.label_mapping.items()}
        with open(label_path, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, ensure_ascii=False, indent=2)
        paths.append(label_path)

        config_path = os.path.join(model_dir, "training_config.yaml")
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                {"params": self.params, "feature_names": self.feature_names},
                fh,
                allow_unicode=True,
            )
        paths.append(config_path)

        logger.info("XGBoost 模型已保存至 %s", model_dir)
        return paths

    def load(self, model_dir: str) -> "XGBoostModel":
        """从目录加载模型与元数据。

        Args:
            model_dir: 模型目录（含 xgboost_model.json 等）。

        Returns:
            加载后的 self。

        Raises:
            FileNotFoundError: 必需文件缺失。
            ImportError: xgboost 未安装。
        """
        model_path = os.path.join(model_dir, "xgboost_model.json")
        feature_path = os.path.join(model_dir, "feature_names.json")
        if not os.path.exists(model_path) or not os.path.exists(feature_path):
            raise FileNotFoundError(f"模型目录缺少必需文件: {model_dir}")

        with open(feature_path, "r", encoding="utf-8") as fh:
            feature_names = json.load(fh)
        self.feature_names = list(feature_names)

        label_path = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as fh:
                mapping = json.load(fh)
            self.label_mapping = {int(k): v for k, v in mapping.items()}

        config_path = os.path.join(model_dir, "training_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}
            if "params" in config:
                self.params = dict(config["params"])

        model_cls = self._model_class()
        model = model_cls(**self.params, verbosity=0)
        model.load_model(model_path)
        self.model = model
        self.is_trained = True
        logger.info("XGBoost 模型已从 %s 加载", model_dir)
        return self

    def _require_trained(self) -> None:
        """检查模型是否已训练。

        Raises:
            RuntimeError: 模型尚未训练。
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit() 或 load()")

    def _prepare_predict_input(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """统一预测输入为按特征列排序的 DataFrame。

        Args:
            X: 特征 DataFrame 或特征矩阵。

        Returns:
            按 FEATURE_VECTOR_COLUMNS 顺序的 DataFrame。
        """
        if isinstance(X, pd.DataFrame):
            return self._check_features(X)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != len(FEATURE_VECTOR_COLUMNS):
            raise ValueError(
                "特征矩阵必须为 (n_samples, %d)" % len(FEATURE_VECTOR_COLUMNS)
            )
        return pd.DataFrame(X_arr, columns=FEATURE_VECTOR_COLUMNS)

    @staticmethod
    def _sklearn_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray,
    ) -> Dict[str, Any]:
        """计算分类评价指标。

        Args:
            y_true: 真实标签。
            y_pred: 预测标签。
            proba: 预测概率矩阵。

        Returns:
            含 accuracy/precision/recall/f1/roc_auc/confusion_matrix 的字典。

        Raises:
            ImportError: scikit-learn 未安装。
        """
        try:
            from sklearn.metrics import (
                accuracy_score,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError("未安装 scikit-learn，无法评估模型") from exc

        report: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        if len(np.unique(y_true)) >= 2 and proba is not None and proba.shape[1] >= 2:
            report["roc_auc"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr")
            )
        return report
