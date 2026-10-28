"""scripts/offline_poc_run.py —— 离线处理 PoC 运行器。

按“小样本验证链路”原则：
1. 从 datasets/poc_input 选取少量视频（DAiSEE 示例 + UTA-RLDD 三类标签各一）；
2. 逐个跑 OfflineVideoProcessor；
3. 调用 offline_poc_report 汇总报告。

用法：
    python scripts/offline_poc_run.py
    python scripts/offline_poc_run.py --input_dir datasets/poc_input --max_frames 300
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import argparse
import json

from src.service.offline_video_processor import OfflineVideoProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 元数据提供方式：标签文件（JSON）或约定文件名
DAISEE_LABEL_FILE = "datasets/labels/daisee_labels.json"
UTA_LABEL_FILE = "datasets/labels/uta_rldd_labels.json"


def _load_label_map(path: str) -> dict:
    """加载标签文件。

    Args:
        path: 标签 JSON 路径。

    Returns:
        视频标识 → 标签字符串映射。
    """
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _infer_meta(video_path: Path, label_map: dict) -> tuple:
    """推断视频的 (source_dataset, subject_id, label)。

    Args:
        video_path: 视频文件。
        label_map: 标签映射。

    Returns:
        (source_dataset, subject_id, label)。
    """
    name = video_path.stem
    if name.startswith("daisee"):
        dataset = "daisee"
        subject_id = name.split("_")[1] if len(name.split("_")) > 1 else name
    else:
        dataset = "uta_rldd"
        subject_id = name.split("_")[1] if len(name.split("_")) > 1 else name
    label = label_map.get(name, "")
    return dataset, subject_id, label


def _iter_candidates(input_dir: Path, max_videos: int) -> list:
    """列出待处理视频并按数据集排序。

    Args:
        input_dir: 输入目录。
        max_videos: 最大处理数量。

    Returns:
        视频文件路径列表。
    """
    videos = sorted(input_dir.glob("*.mp4")) + sorted(input_dir.glob("*.avi"))
    return videos[:max_videos]


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="离线处理 PoC 运行器")
    parser.add_argument("--input_dir", default="datasets/poc_input")
    parser.add_argument("--output_dir", default="datasets/poc_output")
    parser.add_argument("--max_frames", type=int, default=180, help="单视频最大处理帧数")
    parser.add_argument("--stride", type=int, default=1, help="抽帧步长")
    parser.add_argument("--max_videos", type=int, default=6)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"输入目录不存在: {input_dir}，请先运行 scripts/sample_videos.py")

    daisee_map = _load_label_map(DAISEE_LABEL_FILE)
    uta_map = _load_label_map(UTA_LABEL_FILE)

    processor = OfflineVideoProcessor(
        output_dir=args.output_dir,
        window_seconds=30,
        max_frames_per_video=args.max_frames,
        frame_stride=args.stride,
    )

    results = []
    for video in _iter_candidates(input_dir, args.max_videos):
        label_map = daisee_map if video.name.startswith("daisee") else uta_map
        dataset, subject_id, label = _infer_meta(video, label_map)
        logger.info("处理视频 %s（%s, label=%s）", video, dataset, label or "(无标签)")
        stats = processor.process_video(
            video_path=str(video),
            video_id=video.stem,
            source_dataset=dataset,
            label=label,
            subject_id=subject_id,
        )
        results.append(stats.__dict__)

    summary_path = Path(args.output_dir) / "reports" / "poc_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 生成人类可读汇总
    from scripts.offline_poc_report import main as report_main
    import sys
    sys.argv = ["offline_poc_report", "--output_dir", args.output_dir]
    report_main()

    logger.info("PoC 完成，共处理 %s 个视频", len(results))


if __name__ == "__main__":
    main()
