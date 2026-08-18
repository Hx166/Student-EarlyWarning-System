"""src/models/lstm/model.py —— LSTM 课堂状态时序预测模型。

依据 docs/05-机器学习/LSTM.md 实现：
- 基于 PyTorch 的单/双层 LSTM，输入为 Feature Engineering 输出的连续特征序列；
- 与 XGBoost 形成「当前状态分类 + 未来趋势预测」双层架构；
- 提供训练、推理、保存与加载接口，输出统一的 LSTMResult。

说明：本模块不负责特征采集与模型统一注册（ModelRegistry）。
torch 为惰性导入，缺失时模块仍可被导入，仅训练/推理时报错。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from src.core.models import LSTMResult
from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 4 类状态标签（与 XGBoost LABEL_MAPPING 保持一致）
LABEL_MAPPING: Dict[int, str] = {
    0: "normal",
    1: "attention_down",
    2: "fatigue",
    3: "high_risk",
}

# 预测类别到趋势描述的映射（简化策略）
_TREND_BY_LABEL: Dict[str, str] = {
    "normal": "stable",
    "attention_down": "declining",
    "fatigue": "declining",
    "high_risk": "deteriorating",
}

# 默认模型参数（LSTM.md 第 8 节推荐值）
DEFAULT_PARAMS: Dict[str, Any] = {
    "hidden_size": 64,
    "num_layers": 2,
    "num_classes": 4,
    "dropout": 0.2,
    "seq_len": 60,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
}


def _create_net(
    torch: Any,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    num_classes: int,
    dropout: float,
) -> Any:
    """构建 LSTM 网络结构（惰性定义，避免模块导入依赖 torch）。

    Args:
        torch: torch 模块。
        input_size: 特征维度。
        hidden_size: 隐层维度。
        num_layers: LSTM 层数。
        num_classes: 类别数。
        dropout: Dropout 比例。

    Returns:
        LSTM 网络实例。
    """

    class _LSTMNet(torch.nn.Module):
        """LSTM + 全连接分类网络。"""

        def __init__(self) -> None:
            """初始化网络结构。"""
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = torch.nn.Linear(hidden_size, num_classes)

        def forward(self, x: Any) -> Any:
            """前向传播。

            Args:
                x: 输入张量，形状 (batch, seq_len, input_size)。

            Returns:
                分类 logits，形状 (batch, num_classes)。
            """
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return _LSTMNet()


class LSTMModel:
    """LSTM 课堂状态时序预测模型。

    Attributes:
        input_size: 输入特征维度。
        hidden_size: 隐层维度。
        num_layers: LSTM 层数。
        num_classes: 类别数。
        dropout: Dropout 比例。
        seq_len: 序列长度。
        batch_size: 训练批大小。
        epochs: 训练轮数。
        learning_rate: 学习率。
        feature_names: 特征列名。
        label_mapping: 类别 id 到语义名称的映射。
        net: PyTorch 网络实例（训练/加载后可用）。
        is_trained: 模型是否已完成训练。
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: int = DEFAULT_PARAMS["hidden_size"],
        num_layers: int = DEFAULT_PARAMS["num_layers"],
        num_classes: int = DEFAULT_PARAMS["num_classes"],
        dropout: float = DEFAULT_PARAMS["dropout"],
        seq_len: int = DEFAULT_PARAMS["seq_len"],
        batch_size: int = DEFAULT_PARAMS["batch_size"],
        epochs: int = DEFAULT_PARAMS["epochs"],
        learning_rate: float = DEFAULT_PARAMS["learning_rate"],
        label_mapping: Optional[Dict[int, str]] = None,
    ) -> None:
        """初始化 LSTM 模型。

        Args:
            input_size: 输入特征维度，为空时取特征向量列数。
            hidden_size: 隐层维度。
            num_layers: LSTM 层数。
            num_classes: 类别数。
            dropout: Dropout 比例。
            seq_len: 序列长度。
            batch_size: 训练批大小。
            epochs: 训练轮数。
            learning_rate: 学习率。
            label_mapping: 类别 id 到名称的映射。
        """
        self.input_size = input_size or len(FEATURE_VECTOR_COLUMNS)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.label_mapping: Dict[int, str] = dict(
            label_mapping if label_mapping is not None else LABEL_MAPPING
        )
        self.feature_names: List[str] = list(FEATURE_VECTOR_COLUMNS)
        self.net: Optional[Any] = None
        self.is_trained: bool = False
        logger.info("LSTMModel 已初始化，seq_len=%s hidden=%s", seq_len, hidden_size)

    @staticmethod
    def _torch() -> Any:
        """返回 torch 模块。

        Returns:
            torch 模块。

        Raises:
            ImportError: torch 未安装时抛出。
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "未安装 torch，无法使用 LSTM 模型，请执行 pip install torch"
            ) from exc
        return torch

    def _ensure_net(self) -> Any:
        """确保网络已构建。

        Returns:
            LSTM 网络实例。
        """
        if self.net is None:
            torch = self._torch()
            self.net = _create_net(
                torch, self.input_size, self.hidden_size,
                self.num_layers, self.num_classes, self.dropout,
            )
        return self.net

    def fit(
        self,
        sequences: Sequence,
        labels: Sequence[int],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> "LSTMModel":
        """训练模型。

        Args:
            sequences: 特征序列，形状 (n_samples, seq_len, input_size)。
            labels: 标签序列（0~3）。
            epochs: 训练轮数（为空用默认值）。
            batch_size: 批大小（为空用默认值）。
            learning_rate: 学习率（为空用默认值）。

        Returns:
            训练后的 self。

        Raises:
            ValueError: 输入维度或标签数量不合法。
            ImportError: torch 未安装。
        """
        torch = self._torch()
        X = np.asarray(sequences, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        if X.ndim != 3 or X.shape[1] != self.seq_len or X.shape[2] != self.input_size:
            raise ValueError(
                "序列必须为 (n_samples, %d, %d)" % (self.seq_len, self.input_size)
            )
        if len(y) != X.shape[0]:
            raise ValueError("序列与标签数量不一致")

        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        lr = learning_rate if learning_rate is not None else self.learning_rate

        net = self._ensure_net()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        n = X.shape[0]
        net.train()
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = torch.tensor(X[idx], dtype=torch.float32)
                yb = torch.tensor(y[idx], dtype=torch.int64)
                optimizer.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            logger.debug("epoch %s/%s loss=%.4f", epoch + 1, epochs, total_loss)
        self.is_trained = True
        logger.info("LSTM 模型训练完成，样本数=%s", n)
        return self

    def predict_proba(self, sequence: Sequence) -> np.ndarray:
        """预测类别概率。

        Args:
            sequence: 特征序列，形状 (seq_len, input_size) 或 (1, seq_len, input_size)。

        Returns:
            形状为 (1, num_classes) 的概率矩阵。

        Raises:
            RuntimeError: 模型未训练。
            ValueError: 序列维度不合法。
            ImportError: torch 未安装。
        """
        self._require_trained()
        X = self._validate_sequence(sequence)
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        torch = self._torch()
        net = self._ensure_net()
        net.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            proba = torch.softmax(net(tensor), dim=-1).cpu().numpy()
        return np.asarray(proba, dtype=float)

    def predict(self, sequence: Sequence) -> LSTMResult:
        """预测未来课堂状态。

        Args:
            sequence: 特征序列，形状 (seq_len, input_size) 或 (1, seq_len, input_size)。

        Returns:
            LSTM 预测结果对象。

        Raises:
            RuntimeError: 模型未训练。
            ValueError: 序列维度不合法。
            ImportError: torch 未安装。
        """
        proba = self.predict_proba(sequence)[0]
        class_id = int(np.argmax(proba))
        label = str(self.label_mapping.get(class_id, class_id))
        probability = float(proba[class_id])
        return LSTMResult(
            prediction=label,
            probability=probability,
            trend=_TREND_BY_LABEL.get(label, "stable"),
        )

    def save(self, model_dir: str) -> List[str]:
        """保存模型与元数据。

        保存内容（LSTM.md 第 11 节）：
        - lstm_model.pth：模型权重与结构配置；
        - feature_names.json：特征列名；
        - label_mapping.json：类别映射；
        - training_config.yaml：训练参数。

        Args:
            model_dir: 保存目录。

        Returns:
            实际写入的文件路径列表。

        Raises:
            ImportError: 需要保存权重但 torch 未安装时抛出。
        """
        os.makedirs(model_dir, exist_ok=True)
        paths: List[str] = []

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
        config = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, allow_unicode=True)
        paths.append(config_path)

        if self.net is not None:
            torch = self._torch()
            checkpoint = {
                "state_dict": self.net.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
                "seq_len": self.seq_len,
            }
            model_path = os.path.join(model_dir, "lstm_model.pth")
            torch.save(checkpoint, model_path)
            paths.append(model_path)

        logger.info("LSTM 模型已保存至 %s", model_dir)
        return paths

    def load(self, model_dir: str) -> "LSTMModel":
        """从目录加载模型与元数据。

        Args:
            model_dir: 模型目录（含 lstm_model.pth 等）。

        Returns:
            加载后的 self。

        Raises:
            FileNotFoundError: 必需文件缺失。
            ImportError: torch 未安装。
        """
        torch = self._torch()

        feature_path = os.path.join(model_dir, "feature_names.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r", encoding="utf-8") as fh:
                self.feature_names = list(json.load(fh))

        label_path = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as fh:
                mapping = json.load(fh)
            self.label_mapping = {int(k): v for k, v in mapping.items()}

        config_path = os.path.join(model_dir, "training_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}
            self.input_size = int(config.get("input_size", self.input_size))
            self.hidden_size = int(config.get("hidden_size", self.hidden_size))
            self.num_layers = int(config.get("num_layers", self.num_layers))
            self.num_classes = int(config.get("num_classes", self.num_classes))
            self.dropout = float(config.get("dropout", self.dropout))
            self.seq_len = int(config.get("seq_len", self.seq_len))

        model_path = os.path.join(model_dir, "lstm_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件缺失: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        self.input_size = int(checkpoint.get("input_size", self.input_size))
        self.hidden_size = int(checkpoint.get("hidden_size", self.hidden_size))
        self.num_layers = int(checkpoint.get("num_layers", self.num_layers))
        self.num_classes = int(checkpoint.get("num_classes", self.num_classes))
        self.dropout = float(checkpoint.get("dropout", self.dropout))
        self.seq_len = int(checkpoint.get("seq_len", self.seq_len))

        self.net = self._ensure_net()
        self.net.load_state_dict(checkpoint["state_dict"])
        self.is_trained = True
        logger.info("LSTM 模型已从 %s 加载", model_dir)
        return self

    def _validate_sequence(self, sequence: Sequence) -> np.ndarray:
        """校验并规整输入序列。

        Args:
            sequence: 特征序列。

        Returns:
            规整后的 ndarray。

        Raises:
            ValueError: 维度不合法。
        """
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim not in (2, 3):
            raise ValueError("序列必须为 (seq_len, input_size) 或 (1, seq_len, input_size)")
        feature_dim = arr.shape[2] if arr.ndim == 3 else arr.shape[1]
        if feature_dim != self.input_size:
            raise ValueError(
                "序列特征维度不匹配: 期望 %d，实际 %d" % (self.input_size, feature_dim)
            )
        return arr

    def _require_trained(self) -> None:
        """检查模型是否已训练。

        Raises:
            RuntimeError: 模型尚未训练。
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 fit() 或 load()")
