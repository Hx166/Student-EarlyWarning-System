"""阶段 C：20 条 DAiSEE 片段级特征提取验证。

工作流依次为 sample、trial、remaining、analyze。抽样文件生成后只复用，
不会在后续阶段重新随机抽样；本脚本不修改 src/，也不接触 Validation/Test。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.features.builder import FeatureBuilder
from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.utils.config import load_config
from src.utils.logger import get_logger, init_logging

logger = get_logger(__name__)
SAMPLE_COLUMNS = ("clip_id", "subject_id", "split", "engagement", "frames", "ext",
                  "face_found_ratio", "path")
FEATURE_OUTPUT_COLUMNS = (
    "clip_id", "subject_id", "split", "engagement", "aggregation", "frames", "seconds",
    "ear_mean", "ear_std", "blink_rate", "perclos", "pitch_mean", "yaw_mean", "roll_mean",
    "fatigue_score", "attention_score", "head_down_duration", "face_lost_duration",
    "face_found_ratio", "elapsed_seconds",
)
COMPARE_COLUMNS = ("clip_id", "feature", "A", "B", "abs_diff")
TIMING_COLUMNS = ("clip_id", "elapsed_seconds", "face_found_ratio", "frames_processed",
                  "processed_seconds", "state_record_count")
STATE_COLUMNS = ("avg_ear", "avg_mar", "avg_pitch", "avg_yaw", "avg_roll", "perclos",
                 "attention_score", "fatigue_score")
MAX_DELTA = 1e-6


def _safe_name(value: str) -> str:
    """将 clip_id 转为 OfflineVideoProcessor 使用的安全文件名。"""
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def sample_clips(manifest: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """按 split/等级固定抽样，并保证至少两条异常帧 MP4。

    Args:
        manifest: 阶段 B manifest。
        seed: 随机种子。

    Returns:
        冻结候选清单与替换记录。
    """
    candidates = manifest[(manifest["split"] == "Train") & manifest["engagement"].notna()].copy()
    candidates = candidates.sort_values("clip_id").reset_index(drop=True)
    rng = random.Random(seed)
    selected_parts: list[pd.DataFrame] = []
    for engagement in range(4):
        level = candidates[candidates["engagement"] == engagement]
        if len(level) < 5:
            raise ValueError(f"Train Engagement {engagement} 可关联片段不足 5 条")
        selected = level.iloc[rng.sample(range(len(level)), 5)].copy()
        selected_parts.append(selected)
    selected = pd.concat(selected_parts, ignore_index=True)
    anomaly_pool = candidates[
        (candidates["ext"].str.lower() == ".mp4")
        & (candidates["frames"].isin([302, 314]))
    ].sort_values("clip_id")
    anomaly_count = int(
        selected["clip_id"].isin(anomaly_pool["clip_id"]).sum()
    )
    needed = max(0, 2 - anomaly_count)
    available = anomaly_pool[~anomaly_pool["clip_id"].isin(selected["clip_id"])]
    if needed > len(available):
        raise ValueError("Train 异常帧 MP4 不足两条")
    forced = available.iloc[rng.sample(range(len(available)), needed)] if needed else available.iloc[0:0]
    replacements: list[dict[str, Any]] = []
    for _, forced_row in forced.iterrows():
        same_level = selected[
            (selected["engagement"] == forced_row["engagement"])
            & (~selected["frames"].isin([302, 314]))
        ].sort_values("clip_id")
        if same_level.empty:
            raise ValueError(f"Engagement {forced_row['engagement']} 无可替换普通片段")
        replacement = same_level.sample(n=1, random_state=rng.randrange(2**31)).iloc[0]
        selected = selected[selected["clip_id"] != replacement["clip_id"]]
        selected = pd.concat([selected, forced_row.to_frame().T], ignore_index=True)
        replacements.append(
            {
                "removed_clip_id": replacement["clip_id"],
                "inserted_clip_id": forced_row["clip_id"],
                "engagement": int(forced_row["engagement"]),
                "frames": int(forced_row["frames"]),
                "reason": "强制纳入异常帧 MP4",
            }
        )
    selected = selected.sort_values(["engagement", "clip_id"]).reset_index(drop=True)
    if selected["clip_id"].nunique() != 20:
        raise ValueError("抽样后 clip_id 不唯一或不是 20 条")
    return selected, replacements


def write_sample(sample: pd.DataFrame, path: Path, notes_path: Path | None = None,
                 replacements: Sequence[dict[str, Any]] = (), blank_ratio: bool = True) -> None:
    """写入冻结抽样文件及抽样说明。"""
    data = sample.copy()
    if blank_ratio:
        data["face_found_ratio"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.loc[:, SAMPLE_COLUMNS].to_csv(path, index=False, encoding="utf-8")
    if notes_path is not None:
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(
            json.dumps({"seed": 42, "replacements": list(replacements)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def create_or_load_sample(
    manifest_path: Path, sample_path: Path, notes_path: Path, seed: int
) -> pd.DataFrame:
    """生成一次抽样文件；之后始终读取冻结版本。"""
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if len(sample) != 20:
            raise ValueError(f"冻结清单应为 20 行，实际 {len(sample)}")
        logger.info("复用冻结抽样清单: %s", sample_path)
        return sample
    manifest = pd.read_csv(manifest_path)
    sample, replacements = sample_clips(manifest, seed)
    write_sample(sample, sample_path, notes_path, replacements)
    logger.info("已生成冻结抽样清单: %s", sample_path)
    return pd.read_csv(sample_path)


def process_clip(
    record: pd.Series, work_dir: Path, processor: Any
) -> dict[str, Any]:
    """完整处理一个片段并校验输出与帧数。"""
    clip_id = str(record["clip_id"])
    started = time.perf_counter()
    stats = processor.process_video(
        video_path=str(record["path"]),
        video_id=clip_id,
        source_dataset="daisee",
        label=str(int(record["engagement"])),
        subject_id=str(record["subject_id"]),
    )
    elapsed = time.perf_counter() - started
    expected_frames = int(record["frames"])
    if stats.frame_count != expected_frames:
        raise RuntimeError(
            f"{clip_id} 未完整解码: processed={stats.frame_count}, expected={expected_frames}"
        )
    safe_id = _safe_name(clip_id)
    required = (
        work_dir / "records" / f"{safe_id}_state.csv",
        work_dir / "features" / f"{safe_id}_features.csv",
        work_dir / "reports" / f"{safe_id}_stats.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"{clip_id} 缺少产物: {missing}")
    features = pd.read_csv(required[1])
    if abs(len(features) - stats.processed_seconds) > 2:
        raise RuntimeError(f"{clip_id} 特征行数 {len(features)} 与秒数 {stats.processed_seconds} 不符")
    behavior_path = work_dir / "records" / f"{safe_id}_behavior.csv"
    if not behavior_path.exists():
        logger.warning("%s 没有行为事件，生产处理器未生成空 behavior CSV", clip_id)
    return {
        "clip_id": clip_id,
        "elapsed_seconds": round(elapsed, 6),
        "face_found_ratio": stats.face_found_ratio,
        "frames_processed": stats.frame_count,
        "processed_seconds": stats.processed_seconds,
        "state_record_count": stats.state_record_count,
    }


def append_timing(path: Path, result: dict[str, Any]) -> None:
    """按 clip_id 更新逐条处理耗时。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame[frame["clip_id"] != result["clip_id"]]
    else:
        frame = pd.DataFrame(columns=TIMING_COLUMNS)
    frame = pd.concat([frame, pd.DataFrame([result])], ignore_index=True)
    frame.loc[:, TIMING_COLUMNS].to_csv(path, index=False, encoding="utf-8")


def run_records(
    sample: pd.DataFrame, work_dir: Path, timing_path: Path, only_first: bool = False,
    skip_completed: bool = False,
) -> int:
    """使用生产离线处理器处理一条或剩余片段。"""
    from src.service.offline_video_processor import OfflineVideoProcessor

    processor = OfflineVideoProcessor(
        output_dir=str(work_dir),
        window_seconds=10,
        max_frames_per_video=None,
        frame_stride=1,
    )
    completed = set()
    if skip_completed and timing_path.exists():
        completed = set(pd.read_csv(timing_path)["clip_id"].astype(str))
    records = sample.iloc[:1] if only_first else sample
    processed = 0
    for index, record in records.iterrows():
        clip_id = str(record["clip_id"])
        if skip_completed and clip_id in completed:
            continue
        logger.info("开始处理 %s/%s: %s", index + 1, len(sample), clip_id)
        result = process_clip(record, work_dir, processor)
        append_timing(timing_path, result)
        processed += 1
        logger.info("完成 %s，耗时 %.3fs，人脸检出率 %.4f",
                    clip_id, result["elapsed_seconds"], result["face_found_ratio"])
    return processed


def independent_features(
    state_records: pd.DataFrame, behavior_records: pd.DataFrame, window_seconds: int = 10
) -> dict[str, float]:
    """用全部记录独立复算 11 个片段级特征。"""
    values: dict[str, float] = {}
    for column in STATE_COLUMNS:
        series = pd.to_numeric(state_records[column], errors="coerce")
        values[column] = float(series.mean() if series.notna().any() else 0.0)
        if column == "avg_ear":
            values["ear_std"] = float(series.std() if series.notna().sum() > 1 else 0.0)
    behavior = behavior_records.copy()
    if behavior.empty:
        behavior = pd.DataFrame(columns=("behavior_type", "duration"))
    durations = pd.to_numeric(behavior.get("duration", 0.0), errors="coerce").fillna(0.0)
    behavior = behavior.assign(duration=durations)
    kinds = behavior["behavior_type"].astype(str)
    blink_count = int((kinds == "blink").sum())
    values["blink_rate"] = blink_count * 60.0 / window_seconds
    values["head_down_duration"] = float(behavior.loc[kinds == "head_down", "duration"].sum())
    values["face_lost_duration"] = float(behavior.loc[kinds.isin(["looking_away", "absent"]), "duration"].sum())
    mapping = {
        "ear_mean": "avg_ear", "perclos": "perclos", "pitch_mean": "avg_pitch",
        "yaw_mean": "avg_yaw", "roll_mean": "avg_roll", "fatigue_score": "fatigue_score",
        "attention_score": "attention_score",
    }
    result = {feature: values[source] for feature, source in mapping.items()}
    result["ear_std"] = values["ear_std"]
    result.update({key: values[key] for key in (
        "blink_rate", "head_down_duration", "face_lost_duration"
    )})
    return {feature: result[feature] for feature in FEATURE_VECTOR_COLUMNS}


def production_features(state_records: pd.DataFrame, behavior_records: pd.DataFrame) -> pd.Series:
    """取 FeatureBuilder(10) 最后一行的生产特征。"""
    result = FeatureBuilder(window_seconds=10).build(
        state_records.to_dict("records"), behavior_records.to_dict("records")
    )
    if result.empty:
        raise ValueError("生产 FeatureBuilder 返回空表")
    return result.iloc[-1][list(FEATURE_VECTOR_COLUMNS)]


def analyze_sample(
    sample: pd.DataFrame, work_dir: Path, timing_path: Path,
    compare_path: Path, features_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比较 A/B 聚合并导出 20 行特征表。"""
    timings = pd.read_csv(timing_path).set_index("clip_id")
    comparison_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    for _, record in sample.iterrows():
        clip_id = str(record["clip_id"])
        safe_id = _safe_name(clip_id)
        state_path = work_dir / "records" / f"{safe_id}_state.csv"
        behavior_path = work_dir / "records" / f"{safe_id}_behavior.csv"
        state = pd.read_csv(state_path)
        behavior = pd.read_csv(behavior_path) if behavior_path.exists() else pd.DataFrame(
            columns=("behavior_type", "duration")
        )
        a_values = production_features(state, behavior)
        b_values = independent_features(state, behavior)
        clip_rows.append({"clip_id": clip_id, "A": a_values, "B": b_values})
        for feature in FEATURE_VECTOR_COLUMNS:
            a = float(a_values[feature])
            b = float(b_values[feature])
            delta = abs(a - b) if np.isfinite(a) and np.isfinite(b) else float("inf")
            comparison_rows.append(
                {"clip_id": clip_id, "feature": feature, "A": a, "B": b, "abs_diff": delta}
            )
    comparison = pd.DataFrame(comparison_rows, columns=COMPARE_COLUMNS)
    max_deltas = comparison.groupby("feature")["abs_diff"].max()
    consistent = int((max_deltas <= MAX_DELTA).sum())
    aggregation = "A"
    output_rows = []
    for row in clip_rows:
        source = row[aggregation]
        meta = sample[sample["clip_id"] == row["clip_id"]].iloc[0]
        timing = timings.loc[row["clip_id"]]
        values = {feature: float(source[feature]) for feature in FEATURE_VECTOR_COLUMNS}
        output_rows.append({
            "clip_id": row["clip_id"], "subject_id": meta["subject_id"],
            "split": meta["split"], "engagement": int(meta["engagement"]),
            "aggregation": aggregation, "frames": int(meta["frames"]),
            "seconds": float(timing["processed_seconds"]),
            **values, "face_found_ratio": float(timing["face_found_ratio"]),
            "elapsed_seconds": float(timing["elapsed_seconds"]),
        })
    features = pd.DataFrame(output_rows, columns=FEATURE_OUTPUT_COLUMNS)
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(compare_path, index=False, encoding="utf-8")
    features.to_csv(features_path, index=False, encoding="utf-8")
    ratios = features.set_index("clip_id")["face_found_ratio"]
    sample["face_found_ratio"] = sample["clip_id"].map(ratios)
    logger.info("A/B 一致列数=%s/11，聚合方式=%s", consistent, aggregation)
    return comparison, features


def _paths(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path]:
    """解析配置与命令行路径。"""
    training = config["training"]
    datasets = PROJECT_ROOT / str(training.get("datasets_dir", "datasets"))
    data_root = Path(args.data_root or training["daisee_root"])
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    return {
        "data_root": data_root,
        "manifest": args.manifest or datasets / "processed" / "daisee_manifest.csv",
        "sample": args.sample_output or datasets / "processed" / "stage_c_sample.csv",
        "compare": args.compare_output or datasets / "processed" / "stage_c_agg_compare.csv",
        "features": args.features_output or datasets / "features" / "stage_c_features.csv",
        "work": args.work_dir or PROJECT_ROOT / "outputs" / "stage_c",
    }


def build_parser() -> argparse.ArgumentParser:
    """构造阶段 C 命令行参数。"""
    parser = argparse.ArgumentParser(description="DAiSEE 阶段 C 特征验证")
    parser.add_argument("--mode", choices=("sample", "trial", "remaining", "analyze", "all"), default="all")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sample-output", type=Path)
    parser.add_argument("--compare-output", type=Path)
    parser.add_argument("--features-output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """执行阶段 C 的指定步骤。"""
    init_logging()
    args = build_parser().parse_args()
    paths = _paths(args, load_config("training"))
    notes = paths["work"] / "sample_notes.json"
    timing = paths["work"] / "stage_c_timings.csv"
    if args.mode in ("sample", "trial", "remaining", "all"):
        sample = create_or_load_sample(paths["manifest"], paths["sample"], notes, args.seed)
    else:
        sample = pd.read_csv(paths["sample"])
    if args.mode == "sample":
        return
    if args.mode == "trial":
        run_records(sample, paths["work"], timing, only_first=True)
    if args.mode in ("remaining", "all"):
        run_records(sample, paths["work"], timing, skip_completed=True)
    if args.mode in ("analyze", "all"):
        comparison, features = analyze_sample(
            sample, paths["work"], timing, paths["compare"], paths["features"]
        )
        write_sample(sample, paths["sample"], blank_ratio=False)
        logger.info("对比 %s 行，特征 %s 行", len(comparison), len(features))


if __name__ == "__main__":
    main()
