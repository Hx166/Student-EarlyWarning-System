"""阶段 D：DAiSEE 8570 条片段的断点续跑特征提取与特征表构建。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.utils.config import load_config
from src.utils.logger import get_logger, init_logging

logger = get_logger(__name__)
FEATURE_CSV_COLUMNS = (
    "clip_id", "subject_id", "split", "engagement", "aggregation", "frames", "seconds",
    *FEATURE_VECTOR_COLUMNS, "face_found_ratio", "elapsed_seconds",
)
TIMING_COLUMNS = (
    "clip_id", "split", "engagement", "frames", "status", "attempts",
    "elapsed_seconds", "error",
)
FAILURE_COLUMNS = TIMING_COLUMNS
STATS_COLUMNS = (
    "feature", "min", "p1", "p50", "p99", "max", "mean", "std",
    "zero_count", "nan_count",
)
EXPECTED_SPLITS = {"Train": 5357, "Validation": 1429, "Test": 1784}
EXPECTED_LEVELS = {0: 61, 1: 440, 2: 4312, 3: 3757}
EXPECTED_ROWS = 8570


def _safe_name(value: str) -> str:
    """生成与 OfflineVideoProcessor 一致的 safe_id。"""
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """先写 .tmp 再原子替换行 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """原子写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def read_completed_row(path: Path) -> dict[str, Any] | None:
    """读取可解析的完成行；损坏行视为未完成。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("clip_id") else None


def pending_records(records: Sequence[dict[str, Any]], rows_dir: Path) -> list[dict[str, Any]]:
    """剔除已有合法完成行的待处理片段。"""
    pending = []
    for record in records:
        path = rows_dir / f"{_safe_name(str(record['clip_id']))}.json"
        if read_completed_row(path) is None:
            pending.append(record)
    return pending


def _extract_once(record: dict[str, Any], temp_root: Path) -> dict[str, Any]:
    """在一个独立临时目录中完整处理一条片段并返回 A 特征。"""
    from src.service.offline_video_processor import OfflineVideoProcessor

    clip_id = str(record["clip_id"])
    safe_id = _safe_name(clip_id)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"daisee_{safe_id}_", dir=str(temp_root)) as temporary:
        work_dir = Path(temporary)
        processor = OfflineVideoProcessor(
            output_dir=str(work_dir),
            window_seconds=10,
            max_frames_per_video=None,
            frame_stride=1,
        )
        stats = processor.process_video(
            video_path=str(record["path"]),
            video_id=clip_id,
            source_dataset="daisee",
            label=str(int(record["engagement"])),
            subject_id=str(record["subject_id"]),
        )
        features_path = work_dir / "features" / f"{safe_id}_features.csv"
        report_path = work_dir / "reports" / f"{safe_id}_stats.json"
        if not features_path.exists() or not report_path.exists():
            raise RuntimeError("缺少 features CSV 或 stats JSON")
        feature_frame = pd.read_csv(features_path)
        if feature_frame.empty:
            raise RuntimeError("features CSV 为空")
        last = feature_frame.iloc[-1]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report["frame_count"]) <= 0:
            raise RuntimeError("完整解码后帧数为 0")
        missing = [name for name in FEATURE_VECTOR_COLUMNS if name not in last.index]
        if missing:
            raise RuntimeError(f"features CSV 缺少列: {missing}")
        row = {
            "clip_id": clip_id,
            "subject_id": str(record["subject_id"]),
            "split": str(record["split"]),
            "engagement": int(record["engagement"]),
            "aggregation": "A",
            "frames": int(report["frame_count"]),
            "manifest_frames": int(record["frames"]),
            "seconds": float(report["processed_seconds"]),
            **{name: float(last[name]) for name in FEATURE_VECTOR_COLUMNS},
            "face_found_ratio": float(report["face_found_ratio"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
    return row


def process_record(record: dict[str, Any], temp_root: str, rows_dir: str) -> dict[str, Any]:
    """处理一条片段，失败重试一次，成功后原子写完成行。"""
    errors: list[str] = []
    total_elapsed = 0.0
    temp_path = Path(temp_root)
    row_path_root = Path(rows_dir)
    clip_id = str(record["clip_id"])
    for attempt in range(1, 3):
        attempt_started = time.perf_counter()
        try:
            row = _extract_once(record, temp_path)
            row["attempts"] = attempt
            row["retry_errors"] = errors
            atomic_write_json(row_path_root / f"{_safe_name(clip_id)}.json", row)
            total_elapsed += float(row["elapsed_seconds"])
            return {
                "ok": True, "clip_id": clip_id, "attempts": attempt,
                "elapsed_seconds": total_elapsed, "retry_errors": errors,
            }
        except Exception as exc:
            total_elapsed += time.perf_counter() - attempt_started
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
    return {
        "ok": False, "clip_id": clip_id, "attempts": 2,
        "elapsed_seconds": total_elapsed, "error": " | ".join(errors),
    }


def write_failures(failures: Iterable[dict[str, Any]], path: Path) -> None:
    """原子写出当前未解决失败。"""
    frame = pd.DataFrame(list(failures), columns=FAILURE_COLUMNS)
    atomic_write_csv(frame, path)


def write_timings(rows_dir: Path, failures: Iterable[dict[str, Any]], path: Path) -> None:
    """由完成行和未解决失败生成逐片段耗时表。"""
    timings: list[dict[str, Any]] = []
    for row_path in sorted(rows_dir.glob("*.json")):
        row = read_completed_row(row_path)
        if row is None:
            continue
        timings.append({
            "clip_id": row["clip_id"], "split": row["split"],
            "engagement": row["engagement"], "frames": row["frames"], "status": "success",
            "attempts": int(row.get("attempts", 1)),
            "elapsed_seconds": float(row["elapsed_seconds"]), "error": "",
        })
    timings.extend(dict(item, status="failed") for item in failures)
    atomic_write_csv(pd.DataFrame(timings, columns=TIMING_COLUMNS), path)


def run_extraction(
    records: Sequence[dict[str, Any]], rows_dir: Path, temp_root: Path,
    failures_path: Path, timings_path: Path, workers: int, limit: int | None,
) -> tuple[int, int, bool]:
    """有界并行执行，超过 1% 失败率后停止调度。"""
    temp_root.mkdir(parents=True, exist_ok=True)
    to_process = list(records[:limit] if limit is not None else records)
    if not to_process:
        write_failures([], failures_path)
        write_timings(rows_dir, [], timings_path)
        return 0, 0, False
    failures_by_clip: dict[str, dict[str, Any]] = {}
    success = failed = 0
    stopped = False
    started = time.perf_counter()
    iterator = iter(to_process)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        for _ in range(min(workers * 2, len(to_process))):
            record = next(iterator, None)
            if record is not None:
                future_map[executor.submit(
                    process_record, record, str(temp_root), str(rows_dir)
                )] = record
        while future_map:
            done, _ = wait(future_map, return_when=FIRST_COMPLETED)
            for future in done:
                record = future_map.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "ok": False, "clip_id": str(record["clip_id"]), "attempts": 2,
                        "elapsed_seconds": 0.0,
                        "error": f"worker failure: {type(exc).__name__}: {exc}",
                    }
                if result["ok"]:
                    success += 1
                    failures_by_clip.pop(result["clip_id"], None)
                    if result.get("retry_errors"):
                        logger.warning("%s 第 2 次尝试成功", result["clip_id"])
                else:
                    failed += 1
                    failures_by_clip[result["clip_id"]] = {
                        "clip_id": result["clip_id"], "split": record["split"],
                        "engagement": record["engagement"], "frames": record["frames"],
                        "attempts": result["attempts"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "error": result["error"],
                    }
                    write_failures(failures_by_clip.values(), failures_path)
                    logger.error("失败 %s: %s", result["clip_id"], result["error"])
                attempted = success + failed
                if attempted % 100 == 0:
                    logger.info(
                        "进度 %s/%s，成功=%s，失败=%s，累计 %.1fs",
                        attempted, len(to_process), success, failed,
                        time.perf_counter() - started,
                    )
                if attempted and failed / attempted > 0.01:
                    stopped = True
                if not stopped:
                    next_record = next(iterator, None)
                    if next_record is not None:
                        future_map[executor.submit(
                            process_record, next_record, str(temp_root), str(rows_dir)
                        )] = next_record
            if stopped:
                for future in future_map:
                    future.cancel()
                break
    unresolved = list(failures_by_clip.values())
    write_failures(unresolved, failures_path)
    write_timings(rows_dir, unresolved, timings_path)
    return success, failed, stopped


def validation_errors(
    frame: pd.DataFrame, manifest: pd.DataFrame, *,
    expected_rows: int = EXPECTED_ROWS,
    expected_splits: dict[str, int] | None = None,
    expected_levels: dict[int, int] | None = None,
) -> list[str]:
    """检查行数、唯一性、特征顺序、缺失和分布。"""
    errors: list[str] = []
    expected_splits = expected_splits or EXPECTED_SPLITS
    expected_levels = expected_levels or EXPECTED_LEVELS
    if len(frame) != expected_rows:
        errors.append(f"行数 {len(frame)} != {expected_rows}")
    if frame["clip_id"].duplicated().any():
        errors.append("clip_id 存在重复")
    if list(frame.columns) != list(FEATURE_CSV_COLUMNS):
        errors.append("输出列顺序不等于规定顺序")
    if not (frame["aggregation"] == "A").all():
        errors.append("aggregation 必须全部为 A")
    actual_splits = frame.groupby("split").size().to_dict()
    actual_levels = frame.groupby("engagement").size().to_dict()
    if actual_splits != expected_splits:
        errors.append(f"split 分布不符: {actual_splits}")
    if actual_levels != expected_levels:
        errors.append(f"engagement 分布不符: {actual_levels}")
    nan_counts = frame.loc[:, FEATURE_VECTOR_COLUMNS].isna().sum()
    if int(nan_counts.sum()):
        errors.append(f"特征 NaN: {nan_counts[nan_counts > 0].to_dict()}")
    expected_frame = manifest[manifest["engagement"].notna()][
        ["clip_id", "subject_id", "split", "engagement", "frames"]
    ].copy()
    expected_frame["engagement"] = expected_frame["engagement"].astype(int)
    joined = frame.merge(
        expected_frame, on="clip_id", how="outer", suffixes=("", "_expected"), indicator=True
    )
    if len(joined) != len(frame) or (joined["_merge"] != "both").any():
        errors.append("clip_id 集合与 manifest 可关联标签集合不一致")
    if not errors:
        for column in ("subject_id", "split", "engagement"):
            if not joined[column].astype(str).equals(joined[f"{column}_expected"].astype(str)):
                errors.append(f"{column} 与 manifest 不一致")
        if (joined["frames"] > joined["frames_expected"]).any():
            errors.append("实际解码帧数大于 manifest 帧数")
    return errors


def feature_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """计算 11 特征分布统计。"""
    rows = []
    for feature in FEATURE_VECTOR_COLUMNS:
        series = pd.to_numeric(frame[feature], errors="coerce")
        rows.append({
            "feature": feature, "min": series.min(), "p1": series.quantile(0.01),
            "p50": series.quantile(0.50), "p99": series.quantile(0.99),
            "max": series.max(), "mean": series.mean(), "std": series.std(),
            "zero_count": int((series == 0).sum()), "nan_count": int(series.isna().sum()),
        })
    return pd.DataFrame(rows, columns=STATS_COLUMNS)


def compare_with_stage_c(features: pd.DataFrame, stage_c_path: Path) -> float:
    """核对 20 条阶段 C A 特征，返回全局最大差异。"""
    stage_c = pd.read_csv(stage_c_path)
    if not (stage_c["aggregation"] == "A").all():
        raise ValueError("阶段 C 特征不是 A 口径")
    joined = features.merge(stage_c, on="clip_id", suffixes=("", "_stage_c"))
    if len(joined) != 20:
        raise ValueError(f"阶段 C 一致性命中 {len(joined)} 条，应为 20")
    deltas = [
        np.abs(joined[name].to_numpy() - joined[f"{name}_stage_c"].to_numpy())
        for name in FEATURE_VECTOR_COLUMNS
    ]
    maximum = float(np.max(np.concatenate(deltas))) if deltas else 0.0
    if maximum > 1e-9:
        raise ValueError(f"阶段 C A 特征不一致，max|Δ|={maximum}")
    return maximum


def merge_and_validate(
    rows_dir: Path, manifest_path: Path, output_path: Path,
    stats_path: Path, stage_c_path: Path | None = None,
    expected_rows: int = EXPECTED_ROWS,
    expected_splits: dict[str, int] | None = None,
    expected_levels: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """合并 JSON 行、执行完整校验并写出特征表与统计。"""
    rows = [row for path in sorted(rows_dir.glob("*.json"))
            if (row := read_completed_row(path)) is not None]
    frame = pd.DataFrame(rows).reindex(columns=FEATURE_CSV_COLUMNS)
    frame["engagement"] = pd.to_numeric(frame["engagement"], errors="coerce")
    frame["frames"] = pd.to_numeric(frame["frames"], errors="coerce")
    split_order = {name: index for index, name in enumerate(EXPECTED_SPLITS)}
    frame = frame.sort_values(
        ["split", "clip_id"], key=lambda col: col.map(split_order) if col.name == "split" else col
    ).reset_index(drop=True)
    manifest = pd.read_csv(manifest_path)
    errors = validation_errors(
        frame, manifest, expected_rows=expected_rows,
        expected_splits=expected_splits, expected_levels=expected_levels,
    )
    if errors:
        raise ValueError("；".join(errors))
    maximum = compare_with_stage_c(frame, stage_c_path) if stage_c_path else 0.0
    stats = feature_stats(frame)
    atomic_write_csv(frame, output_path)
    atomic_write_csv(stats, stats_path)
    return frame, stats, maximum


def _paths(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path]:
    """从配置和参数解析路径。"""
    training = config["training"]
    datasets = PROJECT_ROOT / str(training.get("datasets_dir", "datasets"))
    data_root = Path(args.data_root or training["daisee_root"])
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    return {
        "data_root": data_root,
        "manifest": args.manifest or datasets / "processed" / "daisee_manifest.csv",
        "output": args.output or datasets / "features" / "daisee_features.csv",
        "stats": args.stats_output or datasets / "processed" / "daisee_features_stats.csv",
        "stage_c": args.stage_c_features or datasets / "features" / "stage_c_features.csv",
        "work": args.work_dir or PROJECT_ROOT / "outputs" / "daisee_extract",
    }


def build_parser() -> argparse.ArgumentParser:
    """构造阶段 D 参数。"""
    parser = argparse.ArgumentParser(description="构建 DAiSEE 全量片段特征表")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--stage-c-features", type=Path)
    parser.add_argument("--work-dir", type=Path)
    return parser


def main() -> None:
    """执行提取、断点续跑、合并与校验。"""
    init_logging()
    args = build_parser().parse_args()
    paths = _paths(args, load_config("training"))
    rows_dir = paths["work"] / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    failures_path = paths["work"] / "failures.csv"
    timing_path = paths["work"] / "timings.csv"
    if args.merge_only:
        merge_and_validate(
            rows_dir, paths["manifest"], paths["output"], paths["stats"], paths["stage_c"]
        )
        return
    manifest = pd.read_csv(paths["manifest"])
    records = manifest[manifest["engagement"].notna()].copy()
    records["engagement"] = records["engagement"].astype(int)
    split_order = {name: index for index, name in enumerate(EXPECTED_SPLITS)}
    records["_split_order"] = records["split"].map(split_order)
    records = records.sort_values(["_split_order", "clip_id"]).drop(
        columns="_split_order"
    ).to_dict("records")
    pending = pending_records(records, rows_dir)
    logger.info("manifest=%s，已完成=%s，待处理=%s", len(records), len(records) - len(pending), len(pending))
    success, failed, stopped = run_extraction(
        pending, rows_dir, paths["work"] / "tmp", failures_path, timing_path,
        max(1, args.workers), args.limit,
    )
    logger.info("本轮成功=%s，失败=%s，停止=%s", success, failed, stopped)
    if stopped:
        raise SystemExit("失败率超过 1%，已停止调度")
    if args.limit is not None:
        return
    frame, stats, maximum = merge_and_validate(
        rows_dir, paths["manifest"], paths["output"], paths["stats"], paths["stage_c"]
    )
    logger.info("全量完成: rows=%s features=%s max_stage_c_delta=%s", len(frame), len(stats), maximum)


if __name__ == "__main__":
    main()
