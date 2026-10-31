"""阶段 E：DAiSEE Engagement 四分类训练、评估与模型注册。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.labels import ENGAGEMENT_LABEL_MAPPING
from src.models.registry import ModelRegistry, register_model_loader
from src.models.xgboost import XGBoostModel
from src.utils.config import load_config
from src.utils.logger import get_logger, init_logging

logger = get_logger(__name__)
DATASET_VERSION = "daisee@2026-10"
SEED = 42
EXPECTED_ROWS = 8570
EXPECTED_SPLITS = {"Train": 5357, "Validation": 1429, "Test": 1784}
EXPECTED_LEVELS = {0: 61, 1: 440, 2: 4312, 3: 3757}


def load_and_validate(path: Path, target: str) -> pd.DataFrame:
    """加载冻结特征表并执行阶段 E 输入校验。"""
    if target != "engagement":
        raise ValueError(f"training.target 必须为 engagement，实际为 {target}")
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"特征行数 {len(frame)} != {EXPECTED_ROWS}")
    if frame["clip_id"].duplicated().any() or frame["clip_id"].isna().any():
        raise ValueError("clip_id 不唯一或缺失")
    missing = [name for name in FEATURE_VECTOR_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")
    features = frame.loc[:, FEATURE_VECTOR_COLUMNS]
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("11 特征存在 NaN 或无穷值")
    levels = set(pd.to_numeric(frame["engagement"], errors="coerce").dropna().astype(int))
    if levels != {0, 1, 2, 3}:
        raise ValueError(f"标签必须仅包含 0~3，实际为 {sorted(levels)}")
    splits = frame.groupby("split").size().to_dict()
    if splits != EXPECTED_SPLITS:
        raise ValueError(f"split 分布不符: {splits}")
    distribution = frame["engagement"].astype(int).value_counts().sort_index().to_dict()
    if distribution != EXPECTED_LEVELS:
        raise ValueError(f"Engagement 分布不符: {distribution}")
    subject_sets = {
        name: set(group["subject_id"].astype(str))
        for name, group in frame.groupby("split")
    }
    if any(subject_sets[a] & subject_sets[b] for a, b in (
        ("Train", "Validation"), ("Train", "Test"), ("Validation", "Test")
    )):
        raise ValueError("Train/Validation/Test 存在跨集被试")
    frame["engagement"] = frame["engagement"].astype(int)
    return frame


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], proba: np.ndarray | None = None
) -> dict[str, Any]:
    """计算准确率、F1、AUC、混淆矩阵和每类指标。"""
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=list(range(4)), zero_division=0
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "roc_auc_ovr": None,
        "confusion_matrix": confusion_matrix(truth, prediction, labels=list(range(4))).tolist(),
        "per_class": [
            {
                "class_id": index,
                "label": ENGAGEMENT_LABEL_MAPPING[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(4)
        ],
    }
    if proba is not None and set(np.unique(truth)) == {0, 1, 2, 3}:
        try:
            result["roc_auc_ovr"] = float(
                roc_auc_score(
                    truth, np.asarray(proba), multi_class="ovr",
                    average="macro", labels=list(range(4))
                )
            )
        except ValueError as exc:
            result["roc_auc_error"] = str(exc)
    else:
        result["roc_auc_error"] = "真实数据未覆盖全部四类或没有概率输出"
    return result


def majority_baseline(y_true: Sequence[int]) -> dict[str, Any]:
    """使用全部预测为类别 2 的多数类基线。"""
    truth = np.asarray(y_true, dtype=int)
    return classification_metrics(truth, np.full(len(truth), 2, dtype=int))


def predict_and_evaluate(model: XGBoostModel, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, Any], np.ndarray]:
    """执行一次模型预测并返回指标和预测标签。"""
    prediction = model.predict(X)
    proba = model.predict_proba(X)
    return classification_metrics(y.to_numpy(), prediction, proba), prediction


def class_weight_values(y: pd.Series) -> np.ndarray:
    """计算 Train 类别频率的平衡样本权重。"""
    labels = y.to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=4)
    return len(labels) / (4.0 * counts[labels])


def feature_importance(model: XGBoostModel) -> pd.DataFrame:
    """按 gain 输出特征重要性。"""
    raw = model.model.get_booster().get_score(importance_type="gain")
    rows = [
        {"feature": name, "gain": float(raw.get(name, raw.get(f"f{index}", 0.0)))}
        for index, name in enumerate(FEATURE_VECTOR_COLUMNS)
    ]
    return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)


def train_ablation(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """用 9 个非规则特征直接训练 XGBClassifier。"""
    from xgboost import XGBClassifier

    columns = [
        name for name in FEATURE_VECTOR_COLUMNS
        if name not in {"fatigue_score", "attention_score"}
    ]
    model = XGBClassifier(**params, verbosity=0)
    model.fit(train[columns], train["engagement"])
    validation_pred = model.predict(validation[columns])
    validation_proba = model.predict_proba(validation[columns])
    test_pred = model.predict(test[columns])
    test_proba = model.predict_proba(test[columns])
    return (
        classification_metrics(validation["engagement"], validation_pred, validation_proba),
        classification_metrics(test["engagement"], test_pred, test_proba),
        columns,
    )


def write_subset(frame: pd.DataFrame, path: Path) -> None:
    """创建父目录并写 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def build_ablation_table(
    model_metrics: dict[str, Any], ablation_metrics: dict[str, Any], n_features: int
) -> pd.DataFrame:
    """构造 full 与 9 特征消融的对照表。"""
    rows = []
    for feature_set, metrics_by_split, columns in (
        ("full_11", model_metrics, len(FEATURE_VECTOR_COLUMNS)),
        ("ablation_9", ablation_metrics, n_features),
    ):
        for split in ("Validation", "Test"):
            full = model_metrics[split]
            current = metrics_by_split[split]
            rows.append({
                "feature_set": feature_set, "split": split, "n_features": columns,
                "accuracy": current["accuracy"], "macro_f1": current["macro_f1"],
                "weighted_f1": current["weighted_f1"], "roc_auc_ovr": current["roc_auc_ovr"],
                "delta_accuracy": current["accuracy"] - full["accuracy"],
                "delta_macro_f1": current["macro_f1"] - full["macro_f1"],
                "delta_weighted_f1": current["weighted_f1"] - full["weighted_f1"],
            })
    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path, metrics: dict[str, Any], test: pd.DataFrame,
    test_pred: np.ndarray, test_proba: np.ndarray, importance: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    """写出阶段 E 的全部指标与预测文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_subset(pd.DataFrame(metrics["test"]["model"]["per_class"]), output_dir / "per_class_test.csv")
    matrix = pd.DataFrame(
        metrics["test"]["model"]["confusion_matrix"], columns=[f"pred_{i}" for i in range(4)]
    )
    matrix.insert(0, "actual", [f"level_{i}" for i in range(4)])
    write_subset(matrix, output_dir / "confusion_test.csv")
    write_subset(importance, output_dir / "feature_importance.csv")
    predictions = pd.DataFrame({
        "clip_id": test["clip_id"].to_numpy(), "y_true": test["engagement"].to_numpy(),
        "y_pred": test_pred,
        **{f"prob_level_{index}": test_proba[:, index] for index in range(4)},
    })
    write_subset(predictions, output_dir / "test_predictions.csv")
    write_subset(ablation, output_dir / "ablation.csv")


def verify_loaded_model(loaded: XGBoostModel, test: pd.DataFrame) -> dict[str, Any]:
    """用三条 Test 样本验证注册表加载后的预测与标签映射。"""
    sample = test[FEATURE_VECTOR_COLUMNS].iloc[:3]
    predictions = loaded.predict(sample)
    labels = [loaded.label_mapping[int(value)] for value in predictions]
    if not all(name.startswith("level_") for name in labels):
        raise RuntimeError(f"加载模型标签映射异常: {loaded.label_mapping}")
    return {
        "sample_count": 3,
        "predictions": predictions.tolist(),
        "labels": labels,
        "label_mapping": loaded.label_mapping,
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """训练主模型、消融模型，评估并保存全部产物。"""
    started = time.perf_counter()
    config = load_config("training")
    training = config["training"]
    params = {**config["xgboost"], "n_jobs": 1, "random_state": SEED}
    datasets = PROJECT_ROOT / str(training.get("datasets_dir", "datasets"))
    features_path = args.features or datasets / "features" / "daisee_features.csv"
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "stage_e"
    models_root = args.models_root or PROJECT_ROOT / str(training.get("model_dir", "models"))
    frame = load_and_validate(features_path, str(training["target"]))
    train = frame[frame["split"] == "Train"].copy()
    validation = frame[frame["split"] == "Validation"].copy()
    test = frame[frame["split"] == "Test"].copy()
    X_train = train[FEATURE_VECTOR_COLUMNS]
    y_train = train["engagement"]

    model = XGBoostModel(params=params, label_mapping=ENGAGEMENT_LABEL_MAPPING)
    fit_started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_started
    validation_metrics, _ = predict_and_evaluate(
        model, validation[FEATURE_VECTOR_COLUMNS], validation["engagement"]
    )
    test_metrics, test_pred = predict_and_evaluate(
        model, test[FEATURE_VECTOR_COLUMNS], test["engagement"]
    )
    test_proba = model.predict_proba(test[FEATURE_VECTOR_COLUMNS])
    baselines = {
        "Validation": majority_baseline(validation["engagement"]),
        "Test": majority_baseline(test["engagement"]),
    }
    weighted: dict[str, Any] = {"enabled": False}
    if test_metrics["accuracy"] <= baselines["Test"]["accuracy"]:
        weighted_model = XGBoostModel(params=params, label_mapping=ENGAGEMENT_LABEL_MAPPING)
        weighted_model.fit(X_train, y_train, sample_weight=class_weight_values(y_train))
        weighted = {
            "enabled": True,
            "Validation": predict_and_evaluate(
                weighted_model, validation[FEATURE_VECTOR_COLUMNS], validation["engagement"]
            )[0],
            "Test": predict_and_evaluate(
                weighted_model, test[FEATURE_VECTOR_COLUMNS], test["engagement"]
            )[0],
        }
    ablation_validation, ablation_test, ablation_features = train_ablation(
        train, validation, test, model.params
    )
    ablation_metrics = {"Validation": ablation_validation, "Test": ablation_test}
    full_metrics = {"Validation": validation_metrics, "Test": test_metrics}
    ablation_table = build_ablation_table(
        full_metrics, ablation_metrics, len(ablation_features)
    )
    importance = feature_importance(model)
    registry = ModelRegistry(models_root=str(models_root))
    model_dir = registry.save_model(
        "xgboost", model, version="engagement-v1",
        feature_names=FEATURE_VECTOR_COLUMNS,
        metadata={
            "dataset": DATASET_VERSION, "rows": len(frame), "params": model.params,
            "label_mapping": ENGAGEMENT_LABEL_MAPPING,
            "test_metrics": test_metrics, "seed": SEED,
        },
    )
    register_model_loader("xgboost", lambda: XGBoostModel())
    loaded_registry = ModelRegistry(models_root=str(models_root))
    loaded = loaded_registry.get_model("xgboost", "engagement-v1")
    if loaded is None:
        raise RuntimeError("保存后的模型无法通过 ModelRegistry 加载")
    load_verification = verify_loaded_model(loaded, test)
    elapsed = time.perf_counter() - started
    metrics = {
        "dataset": DATASET_VERSION, "rows": len(frame),
        "splits": frame.groupby("split").size().to_dict(),
        "engagement": frame["engagement"].value_counts().sort_index().to_dict(),
        "params": model.params, "seed": SEED, "target": "engagement",
        "fit_seconds": fit_seconds, "elapsed_seconds": elapsed,
        "validation": {"baseline": baselines["Validation"], "model": validation_metrics},
        "test": {"baseline": baselines["Test"], "model": test_metrics},
        "weighted": weighted,
        "ablation": {
            "excluded": ["fatigue_score", "attention_score"],
            "features": ablation_features, "metrics": ablation_metrics,
        },
        "feature_importance_gain": importance.head(10).to_dict("records"),
        "model_dir": str(model_dir),
        "load_verification": load_verification,
    }
    save_outputs(output_dir, metrics, test, test_pred, test_proba, importance, ablation_table)
    (Path(model_dir) / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("训练完成：model=%s elapsed=%.3fs", model_dir, elapsed)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """构造训练脚本参数。"""
    parser = argparse.ArgumentParser(description="训练 DAiSEE Engagement 四分类模型")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models-root", type=Path)
    return parser


def main() -> None:
    """命令行入口。"""
    init_logging()
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
