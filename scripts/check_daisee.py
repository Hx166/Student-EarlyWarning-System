"""DAiSEE 阶段 B 数据核查与 manifest 构建脚本。

本脚本只读取视频元数据，并对每个 split 随机抽取少量视频读取一帧。
它不执行全量帧解码、MediaPipe、特征提取或模型训练，也不写入原始数据目录。

示例：
    python scripts/check_daisee.py --data-root "D:/数据集/DAiSEE/DAiSEE"
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger

SPLITS = ("Train", "Validation", "Test")
VIDEO_EXTENSIONS = {".avi", ".mp4"}
LABEL_COLUMNS = ("Boredom", "Confusion", "Engagement", "Frustration")
MANIFEST_FIELDS = (
    "split", "subject_id", "clip_id", "path", "ext", "size_mb", "fps", "width",
    "height", "frames", "duration", "boredom", "confusion", "engagement", "frustration", "decodable",
)
ISSUE_FIELDS = ("split", "subject_id", "clip_id", "path", "issue_type", "severity", "detail")

logger = get_logger(__name__)


@dataclass(frozen=True)
class LabelRecord:
    """一条规范化后的 DAiSEE 标签记录。"""

    split: str
    clip_id: str
    boredom: int
    confusion: int
    engagement: int
    frustration: int


@dataclass
class VideoRecord:
    """一个视频文件的关联信息、元数据和抽样解码状态。"""

    split: str
    subject_id: str
    clip_id: str
    path: str
    ext: str
    size_mb: float
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    duration: float | None = None
    metadata_ok: bool = True
    sample_checked: bool = False
    sample_ok: bool | None = None
    sample_error: str = ""
    label: LabelRecord | None = None

    @property
    def decodable(self) -> bool:
        """返回元数据可打开且未在抽样 read() 中失败的状态。"""
        return self.metadata_ok and self.sample_ok is not False


def _issue(split: str, clip_id: str, issue_type: str, detail: str, *, subject_id: str = "",
           path: str = "", severity: str = "warning") -> dict[str, str]:
    """构造统一格式的异常记录。"""
    return {"split": split, "subject_id": subject_id, "clip_id": clip_id, "path": path,
            "issue_type": issue_type, "severity": severity, "detail": detail}


def _parse_level(column: str, raw_value: str) -> int:
    """解析并校验 0 到 3 的 DAiSEE 等级。"""
    value = int(str(raw_value).strip())
    if value < 0 or value > 3:
        raise ValueError(f"{column} 越界: {value}")
    return value


def load_labels(label_path: Path, split: str) -> tuple[dict[str, LabelRecord], list[dict[str, str]]]:
    """读取标签 CSV，规范化表头并按 ClipID 返回映射与异常。"""
    labels: dict[str, LabelRecord] = {}
    issues: list[dict[str, str]] = []
    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        required = {"ClipID", *LABEL_COLUMNS}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{label_path} 缺少列: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            clip_id = str(row.get("ClipID") or "").strip()
            if not clip_id:
                issues.append(
                    _issue(split, "", "invalid_label_row", f"第 {line_number} 行 ClipID 为空", severity="error")
                )
                continue
            try:
                values = {column.lower(): _parse_level(column, row.get(column, "")) for column in LABEL_COLUMNS}
            except (TypeError, ValueError) as exc:
                issues.append(
                    _issue(split, clip_id, "invalid_label_value", f"第 {line_number} 行: {exc}", severity="error")
                )
                continue
            if clip_id in labels:
                issues.append(
                    _issue(split, clip_id, "duplicate_label", f"第 {line_number} 行重复 ClipID", severity="error")
                )
                continue
            labels[clip_id] = LabelRecord(split=split, clip_id=clip_id, **values)
    return labels, issues


def discover_videos(data_root: Path) -> list[VideoRecord]:
    """扫描三个 split，建立不读取帧的视频记录。"""
    records: list[VideoRecord] = []
    for split in SPLITS:
        split_dir = data_root / "DataSet" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"缺少 split 目录: {split_dir}")
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            stat = path.stat()
            records.append(
                VideoRecord(
                    split=split,
                    subject_id=path.parent.parent.name,
                    clip_id=path.name.strip(),
                    path=str(path.resolve()),
                    ext=path.suffix.lower(),
                    size_mb=round(stat.st_size / (1024 * 1024), 3),
                )
            )
    return records


def probe_metadata(record: VideoRecord) -> tuple[VideoRecord, str, str]:
    """仅通过 OpenCV 属性读取视频头信息，不调用 read()。"""
    capture = cv2.VideoCapture(record.path)
    try:
        if not capture.isOpened():
            record.metadata_ok = False
            return record, "OpenCV 无法打开视频", ""
        fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
        width_raw = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_raw = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        invalid = [
            name
            for name, value in (
                ("fps", fps_raw),
                ("width", width_raw),
                ("height", height_raw),
                ("frames", frames_raw),
            )
            if not math.isfinite(value) or value <= 0
        ]
        record.fps = round(fps_raw, 3) if fps_raw > 0 and math.isfinite(fps_raw) else None
        record.width = int(round(width_raw)) if width_raw > 0 and math.isfinite(width_raw) else None
        record.height = int(round(height_raw)) if height_raw > 0 and math.isfinite(height_raw) else None
        record.frames = int(round(frames_raw)) if frames_raw > 0 and math.isfinite(frames_raw) else None
        if record.fps and record.frames:
            record.duration = round(record.frames / record.fps, 3)
        deviations = [name for name, actual, common in (
            ("fps", record.fps, 30), ("width", record.width, 640),
            ("height", record.height, 480), ("frames", record.frames, 300)) if actual != common]
        if invalid:
            record.metadata_ok = False
            return record, "", f"无效元数据字段: {', '.join(invalid)}"
        return record, "", f"偏离常见规格: {', '.join(deviations)}" if deviations else ""
    except Exception as exc:
        record.metadata_ok = False
        return record, f"OpenCV 元数据异常: {exc}", ""
    finally:
        capture.release()


def probe_all_metadata(
    records: Sequence[VideoRecord], workers: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """并行读取全量视频元数据。

    Args:
        records: 视频记录。
        workers: 并行线程数。

    Returns:
        打开失败异常与元数据异常。
    """
    failures: list[dict[str, str]] = []
    anomalies: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [(record, executor.submit(probe_metadata, record)) for record in records]
        for index, (record, future) in enumerate(futures, start=1):
            checked, open_error, anomaly = future.result()
            if open_error:
                failures.append(
                    _issue(
                        checked.split,
                        checked.clip_id,
                        "metadata_open_failure",
                        open_error,
                        subject_id=checked.subject_id,
                        path=checked.path,
                        severity="error",
                    )
                )
            if anomaly:
                anomalies.append(
                    _issue(
                        checked.split,
                        checked.clip_id,
                        "metadata_anomaly",
                        anomaly,
                        subject_id=checked.subject_id,
                        path=checked.path,
                        severity="error" if not checked.metadata_ok else "warning",
                    )
                )
            if index % 1000 == 0:
                logger.info("元数据探测进度: %s/%s", index, len(records))
    return failures, anomalies


def read_first_frame(record: VideoRecord) -> VideoRecord:
    """对一个视频实际读取一帧并记录结果。"""
    capture = cv2.VideoCapture(record.path)
    try:
        if not capture.isOpened():
            record.sample_ok = False
            record.sample_error = "OpenCV 无法打开视频以执行首帧 read()"
            return record
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            record.sample_ok = False
            record.sample_error = "首帧 read() 返回 False 或空帧"
            return record
        record.sample_ok = True
        return record
    except Exception as exc:
        record.sample_ok = False
        record.sample_error = f"首帧 read() 异常: {exc}"
        return record
    finally:
        record.sample_checked = True
        capture.release()


def sample_decodability(
    records: Sequence[VideoRecord], sample_size: int, seed: int
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    """每个 split 固定随机抽取若干视频并只读取一帧。

    Args:
        records: 全部视频记录。
        sample_size: 每个 split 的抽样数量。
        seed: 随机种子。

    Returns:
        抽样解码失败异常和每 split 汇总。
    """
    rng = random.Random(seed)
    failures: list[dict[str, str]] = []
    summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        candidates = [record for record in records if record.split == split]
        selected = rng.sample(candidates, min(sample_size, len(candidates)))
        successes = 0
        for record in selected:
            read_first_frame(record)
            if record.sample_ok:
                successes += 1
            else:
                failures.append(
                    _issue(
                        split,
                        record.clip_id,
                        "decode_failure_sample",
                        record.sample_error,
                        subject_id=record.subject_id,
                        path=record.path,
                        severity="error",
                    )
                )
        summary[split] = {"attempted": len(selected), "success": successes, "failed": len(selected) - successes}
        logger.info("抽样解码 %s: 成功 %s/%s", split, successes, len(selected))
    return failures, summary


def associate_labels(
    records: Sequence[VideoRecord], labels_by_split: dict[str, dict[str, LabelRecord]]
) -> list[dict[str, str]]:
    """关联视频与标签，并记录无标签视频和孤立标签。"""
    issues: list[dict[str, str]] = []
    for record in records:
        record.label = labels_by_split[record.split].get(record.clip_id)
        if record.label is None:
            issues.append(
                _issue(
                    record.split,
                    record.clip_id,
                    "unlabeled_video",
                    "标签 CSV 中不存在对应 ClipID",
                    subject_id=record.subject_id,
                    path=record.path,
                )
            )
    for split in SPLITS:
        videos = {record.clip_id for record in records if record.split == split}
        for label in labels_by_split[split].values():
            if label.clip_id not in videos:
                issues.append(
                    _issue(
                        split,
                        label.clip_id,
                        "orphan_label",
                        "标签记录无对应视频文件",
                        subject_id=label.clip_id[:6],
                        severity="error",
                    )
                )
    return issues


def build_manifest_rows(records: Iterable[VideoRecord]) -> list[dict[str, Any]]:
    """把视频记录转换为 manifest CSV 行。"""
    rows: list[dict[str, Any]] = []
    for record in records:
        label = record.label
        rows.append(
            {
                "split": record.split,
                "subject_id": record.subject_id,
                "clip_id": record.clip_id,
                "path": record.path,
                "ext": record.ext,
                "size_mb": record.size_mb,
                "fps": record.fps or "",
                "width": record.width or "",
                "height": record.height or "",
                "frames": record.frames or "",
                "duration": record.duration or "",
                "boredom": label.boredom if label else "",
                "confusion": label.confusion if label else "",
                "engagement": label.engagement if label else "",
                "frustration": label.frustration if label else "",
                "decodable": "true" if record.decodable else "false",
            }
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    """写出 UTF-8 CSV，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _config_path_value(config: dict[str, Any], keys: Sequence[str]) -> Path | None:
    """从嵌套配置读取第一个非空路径。"""
    for key in keys:
        value: Any = config
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value:
            path = Path(str(value))
            return path if path.is_absolute() else PROJECT_ROOT / path
    return None


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="DAiSEE 数据验证与 manifest 构建")
    parser.add_argument("--data-root", type=Path, help="DAiSEE 根目录；也可配置 training.daisee_root")
    parser.add_argument("--manifest-output", type=Path, help="manifest CSV 输出路径")
    parser.add_argument("--issues-output", type=Path, help="issues CSV 输出路径")
    parser.add_argument("--sample-size", type=int, default=20, help="每个 split 抽样视频数")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--workers", type=int, default=8, help="元数据探测线程数")
    return parser


def _resolve_paths(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """根据命令行与 training.yaml 解析输入输出路径。"""
    data_root = args.data_root or _config_path_value(
        config, ("training.daisee_root", "training.dataset_root", "training.data_root")
    )
    if data_root is None:
        raise ValueError("未提供 --data-root，且 training.yaml 未配置 training.daisee_root")
    datasets_dir = _config_path_value(config, ("training.datasets_dir",)) or PROJECT_ROOT / "datasets"
    manifest = args.manifest_output or datasets_dir / "processed" / "daisee_manifest.csv"
    issues = args.issues_output or datasets_dir / "processed" / "daisee_issues.csv"
    return Path(data_root), Path(manifest), Path(issues)


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    """执行完整核查并写出 manifest 与异常清单。"""
    started = time.perf_counter()
    config = load_config("training")
    data_root, manifest_path, issues_path = _resolve_paths(args, config)
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")

    records = discover_videos(data_root)
    logger.info("发现视频 %s 个，开始读取元数据", len(records))
    labels_by_split: dict[str, dict[str, LabelRecord]] = {}
    issues: list[dict[str, str]] = []
    for split in SPLITS:
        labels, label_issues = load_labels(data_root / "Labels" / f"{split}Labels.csv", split)
        labels_by_split[split] = labels
        issues.extend(label_issues)

    metadata_failures, metadata_anomalies = probe_all_metadata(records, args.workers)
    issues.extend(metadata_failures)
    issues.extend(metadata_anomalies)

    duplicate_videos = Counter(record.clip_id for record in records)
    for record in records:
        if duplicate_videos[record.clip_id] > 1:
            issues.append(
                _issue(
                    record.split,
                    record.clip_id,
                    "duplicate_video",
                    "同一 ClipID 在视频目录中重复出现",
                    subject_id=record.subject_id,
                    path=record.path,
                    severity="error",
                )
            )
    issues.extend(associate_labels(records, labels_by_split))
    decode_failures, sample_summary = sample_decodability(records, args.sample_size, args.seed)
    issues.extend(decode_failures)

    issues.sort(key=lambda item: (SPLITS.index(item["split"]), item["issue_type"], item["clip_id"]))
    write_csv(manifest_path, build_manifest_rows(records), MANIFEST_FIELDS)
    write_csv(issues_path, issues, ISSUE_FIELDS)

    elapsed = time.perf_counter() - started
    logger.info(
        "核查完成: videos=%s labels=%s unlabeled=%s orphans=%s elapsed=%.2fs",
        len(records),
        sum(len(items) for items in labels_by_split.values()),
        sum(record.label is None for record in records),
        sum(issue["issue_type"] == "orphan_label" for issue in issues),
        elapsed,
    )
    return {
        "videos": len(records),
        "labels": sum(len(items) for items in labels_by_split.values()),
        "unlabeled": sum(record.label is None for record in records),
        "orphans": sum(issue["issue_type"] == "orphan_label" for issue in issues),
        "sample_summary": sample_summary,
        "elapsed_seconds": elapsed,
        "manifest": manifest_path,
        "issues": issues_path,
    }


def main() -> None:
    """命令行入口。"""
    args = _build_parser().parse_args()
    try:
        run_check(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
