"""阶段 F：训练侧、在线侧与冻结特征表的三方一致性验证。"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.features.builder import FeatureBuilder
from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.service.prediction import PredictionService
from src.utils.config import load_config
from src.utils.logger import get_logger, init_logging

logger = get_logger(__name__)
TOLERANCE = 1e-9


def _behavior_frame(path: Path) -> pd.DataFrame:
    """读取行为记录；缺失时返回空表。"""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=("behavior_type", "duration"))


def verify_clip(
    clip: pd.Series, feature_table: pd.DataFrame, output_rows: list[dict],
) -> float:
    """处理一个片段并追加三方逐特征对比。"""
    from src.service.offline_video_processor import OfflineVideoProcessor

    clip_id = str(clip["clip_id"])
    safe_id = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in clip_id
    )
    with tempfile.TemporaryDirectory(prefix=f"parity_{safe_id}_") as temporary:
        processor = OfflineVideoProcessor(output_dir=temporary, window_seconds=10)
        processor.process_video(
            video_path=str(clip["path"]),
            video_id=clip_id,
            source_dataset="daisee",
            label=str(int(clip["engagement"])),
            subject_id=str(clip["subject_id"]),
        )
        root = Path(temporary)
        state = pd.read_csv(root / "records" / f"{safe_id}_state.csv")
        behavior = _behavior_frame(root / "records" / f"{safe_id}_behavior.csv")
        behavior_records = behavior.to_dict("records")
        training_row = FeatureBuilder(window_seconds=10).build(
            state.to_dict("records"), behavior_records
        ).iloc[-1]

        service = PredictionService(window_seconds=10, min_records=10)
        for record in state.to_dict("records"):
            service.predict(record, behavior_records=behavior_records)
        online_row = service._builder.build(
            list(service._buffer), behavior_records
        ).iloc[-1]

    table_row = feature_table.loc[
        feature_table["clip_id"] == clip_id
    ].iloc[0]
    max_delta = 0.0
    for feature in FEATURE_VECTOR_COLUMNS:
        training = float(training_row[feature])
        online = float(online_row[feature])
        frozen = float(table_row[feature])
        delta = max(abs(training - frozen), abs(online - frozen), abs(training - online))
        max_delta = max(max_delta, delta)
        output_rows.append(
            {
                "clip_id": clip_id,
                "feature": feature,
                "training_side": training,
                "online_side": online,
                "feature_table": frozen,
                "abs_diff": delta,
            }
        )
    return max_delta


def build_parser() -> argparse.ArgumentParser:
    """构造一致性验证参数。"""
    parser = argparse.ArgumentParser(description="验证 DAiSEE 训练/在线特征一致性")
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """执行前三条片段的特征一致性验证。"""
    init_logging()
    args = build_parser().parse_args()
    training = load_config("training")
    datasets = PROJECT_ROOT / str(training.get("datasets_dir", "datasets"))
    sample_path = args.sample or datasets / "processed" / "stage_c_sample.csv"
    features_path = args.features or datasets / "features" / "daisee_features.csv"
    output_path = args.output or PROJECT_ROOT / "outputs" / "stage_f" / "feature_parity.csv"
    sample = pd.read_csv(sample_path).head(3)
    feature_table = pd.read_csv(features_path)
    rows: list[dict] = []
    clip_deltas = {
        str(row["clip_id"]): verify_clip(row, feature_table, rows)
        for _, row in sample.iterrows()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    maximum = max(clip_deltas.values())
    for clip_id, delta in clip_deltas.items():
        logger.info("clip=%s max_abs_diff=%.12g", clip_id, delta)
    result = "PASS" if maximum <= TOLERANCE else "FAIL"
    logger.info("%s max_abs_diff=%.12g threshold=%.1e", result, maximum, TOLERANCE)
    if result == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
