"""scripts/offline_poc_report.py —— 离线处理 PoC 汇总报告。

汇总 datasets/poc_output/records 与 features 下的结果，
输出：视频实际读取参数、Face Mesh 成功率、face_lost 比例、
EAR/blink/PERCLOS/姿态分布、疲劳/参与度分布、11 维特征统计、
标签映射统计、窗口数量与重叠分析、跨数据集分布对比。

用法：
    python scripts/offline_poc_report.py --output_dir datasets/poc_output
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 11 维特征中由行为/窗口生成、可能大量为 0 的列
ZERO_SENSITIVE = [
    "blink_rate",
    "head_down_duration",
    "face_lost_duration",
]

# DAiSEE 标签映射（Engagement 0/1/2/3 → 训练标签）
DAISEE_LABEL_MAP = {0: "normal", 1: "normal", 2: "attention_down", 3: "attention_down"}

# UTA-RLDD 标签映射
UTA_LABEL_MAP = {
    "alert": "normal",
    "low_vigilant": "attention_down",
    "drowsy": "fatigue",
}


def _percentiles(series: pd.Series) -> dict:
    """计算序列统计描述。

    Args:
        series: 数值序列。

    Returns:
        描述字典。
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    q = s.quantile([0.25, 0.5, 0.75]).to_dict()
    return {
        "count": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4),
        "p25": round(float(q.get(0.25, 0.0)), 4),
        "median": round(float(q.get(0.5, 0.0)), 4),
        "p75": round(float(q.get(0.75, 0.0)), 4),
        "max": round(float(s.max()), 4),
    }


def _load_stats(output_dir: Path) -> list:
    """加载全部视频统计 JSON。

    Args:
        output_dir: 输出目录根。

    Returns:
        统计字典列表。
    """
    stats = []
    reports_dir = output_dir / "reports"
    if reports_dir.exists():
        for p in sorted(reports_dir.glob("*_stats.json")):
            stats.append(json.loads(p.read_text(encoding="utf-8")))
    return stats


def _load_features(output_dir: Path) -> pd.DataFrame:
    """加载全部特征 CSV 并合并。

    Args:
        output_dir: 输出目录根。

    Returns:
        合并后的特征 DataFrame。
    """
    frames = []
    features_dir = output_dir / "features"
    if features_dir.exists():
        for p in sorted(features_dir.glob("*_features.csv")):
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _analyze_nan_zero(df: pd.DataFrame) -> pd.DataFrame:
    """分析特征列的 NaN 与全零情况。

    Args:
        df: 特征 DataFrame。

    Returns:
        每列 NaN 比例与零值比例。
    """
    if df.empty:
        return pd.DataFrame()
    rows = {"feature": [], "nan_ratio": [], "zero_ratio": []}
    for col in FEATURE_VECTOR_COLUMNS:
        if col not in df.columns:
            rows["feature"].append(col)
            rows["nan_ratio"].append(None)
            rows["zero_ratio"].append(None)
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows["feature"].append(col)
        rows["nan_ratio"].append(round(float(s.isna().mean()), 4))
        rows["zero_ratio"].append(round(float((s.fillna(0.0) == 0.0).mean()), 4))
    return pd.DataFrame(rows)


def _window_overlap_analysis(df: pd.DataFrame) -> dict:
    """估算 30 秒滑动窗口样本之间的重叠。

    Args:
        df: 特征 DataFrame（含 record_time）。

    Returns:
        重叠分析结果。
    """
    if df.empty or "record_time" not in df.columns:
        return {}
    diffs: list = []
    per_video: dict = {}
    for vid, sub in df.groupby("video_id") if "video_id" in df.columns else [("all", df)]:
        times = pd.to_datetime(sub["record_time"], errors="coerce").sort_values()
        shifted = times - times.shift(1)
        d = shifted.dropna().dt.total_seconds()
        diffs.extend(d.tolist())
        per_video[vid] = {
            "samples": int(len(sub)),
            "range_seconds": round(float((times.max() - times.min()).total_seconds()), 2) if len(times) else 0.0,
        }
    return {
        "total_samples": int(len(df)),
        "unique_videos": int(df["video_id"].nunique()) if "video_id" in df.columns else 0,
        "median_window_step_seconds": round(float(pd.Series(diffs).median()), 2) if diffs else None,
        "min_window_step_seconds": round(float(pd.Series(diffs).min()), 2) if diffs else None,
        "max_window_step_seconds": round(float(pd.Series(diffs).max()), 2) if diffs else None,
        "per_video": per_video,
        "note": "窗口步长约 1 秒 => 相邻 30 秒窗口高度重叠；DAiSEE 每段仅 10 秒，落在半窗口区（永远无法形成完整 30 秒窗口）。",
    }


def _domain_shift(df: pd.DataFrame) -> pd.DataFrame:
    """对比不同数据集在特征上的均值差异。

    Args:
        df: 特征 DataFrame（含 source_dataset）。

    Returns:
        数据集 × 特征均值对比表。
    """
    if df.empty or "source_dataset" not in df.columns:
        return pd.DataFrame()
    groups = []
    for ds, sub in df.groupby("source_dataset"):
        row = {"source_dataset": ds, "n": len(sub)}
        for col in FEATURE_VECTOR_COLUMNS:
            row[col] = round(float(pd.to_numeric(sub[col], errors="coerce").mean()), 4)
        groups.append(row)
    return pd.DataFrame(groups)


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="离线处理 PoC 汇总报告")
    parser.add_argument("--output_dir", default="datasets/poc_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    stats = _load_stats(output_dir)
    features = _load_features(output_dir)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("离线处理 PoC 汇总报告")
    lines.append("=" * 70)

    # 1. 视频实际读取参数与 Face Mesh 成功率
    lines.append("\n## 1. 视频实际读取参数")
    if stats:
        for s in stats:
            lines.append(
                f"- {s['source_dataset']} | {s['video_id']} | {s['label']} | "
                f"fps={s['fps']} | {s['width']}x{s['height']} | 帧数={s['frame_count']} | "
                f"时长(处理)={s['processed_seconds']}s | 秒记录={s['state_record_count']}"
            )
    else:
        lines.append("- 无统计结果（先运行 offline_poc_run）")

    lines.append("\n## 2. Face Mesh 成功率 / face_lost")
    for s in stats:
        lines.append(
            f"- {s['video_id']}: 检出率={s['face_found_ratio']:.3f} | "
            f"face_lost={s['face_lost_ratio']:.3f} | 持续时长={s['face_lost_duration']}s "
            f"| blob={s['blink_total']} head_down={s['head_down_total']} yawn={s['yawn_total']}"
        )

    # 3-6. 分布统计
    if not features.empty:
        lines.append("\n## 3-6. 特征分布（11 维 + 疲劳/参与度）")
        for col in FEATURE_VECTOR_COLUMNS:
            if col in features.columns:
                p = _percentiles(features[col])
                lines.append(
                    f"- {col}: n={p['count']} mean={p['mean']} std={p['std']} "
                    f"min={p['min']} p50={p['median']} max={p['max']}"
                )
        lines.append("\n## 7. NaN / 零值检查")
        nan_zero = _analyze_nan_zero(features)
        for _, row in nan_zero.iterrows():
            lines.append(f"- {row['feature']}: NaN={row['nan_ratio']} zero={row['zero_ratio']}")

        lines.append("\n## 9. 标签映射后的结果")
        for ds, sub in features.groupby("source_dataset"):
            lines.append(f"- {ds}: {dict(Counter(sub['label']))}")

        lines.append("\n## 10. 训练窗口数量与重叠")
        overlap = _window_overlap_analysis(features)
        for k, v in overlap.items():
            lines.append(f"- {k}: {v}")

        lines.append("\n## 11. 跨数据集 Domain Shift（特征均值）")
        shift = _domain_shift(features)
        if not shift.empty:
            for _, row in shift.iterrows():
                lines.append("- " + " | ".join(f"{c}={row[c]}" for c in row.index))

    report = "\n".join(lines)
    report_path = output_dir / "reports" / "poc_summary.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    logger.info("汇总报告已生成: %s", report_path)


if __name__ == "__main__":
    main()
