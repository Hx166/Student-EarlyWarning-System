"""scripts/sample_videos.py —— 从原始数据集抽样生成 PoC 模拟视频。

不下载数据集，而是从用户已放置（或后续放置）的原始视频中抽样，
生成统一的 480p、固定 FPS 的离线处理测试视频；若对应原始视频缺失，
则按数据集语义生成合成视频（人脸/低参与/低警觉/欲睡），保证离线
链路可在无真实下载时先行验证。

用法：
    python scripts/sample_videos.py --out_dir datasets/poc_input
    python scripts/sample_videos.py --out_dir datasets/poc_input --raw_dir <原始视频目录>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 各数据集抽样规格（来源数据集, 视频标识, 标签）
DAISEE_SAMPLES = [
    ("daisee", "daisee_s001_001", "normal"),
    ("daisee", "daisee_s002_002", "attention_down"),
]

UTA_SAMPLES = [
    ("uta_rldd", "uta_s001_alert", "alert"),
    ("uta_rldd", "uta_s002_lowvigilant", "low_vigilant"),
    ("uta_rldd", "uta_s003_drowsy", "drowsy"),
]

OUT_WIDTH = 640
OUT_HEIGHT = 480
OUT_FPS = 30.0
DAISEE_SECONDS = 10
UTA_SECONDS = 15


def _make_synthetic(video_id: str, label: str, seconds: float, fps: float) -> None:
    """生成合成模拟视频。

    Args:
        video_id: 视频标识。
        label: 标签。
        seconds: 时长（秒）。
        fps: 帧率。
    """
    h, w = OUT_HEIGHT, OUT_WIDTH
    total = int(seconds * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = Path("datasets/poc_input")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.mp4"
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {out_path}")

    rng = np.random.default_rng(abs(hash(video_id)) % (2 ** 32))
    fatigue = "drowsy" in label
    low_attn = "down" in label or "vigilant" in label or "drowsy" in label
    for i in range(total):
        t = i / fps
        frame = np.full((h, w, 3), (220, 220, 230), dtype=np.uint8)
        cv2.rectangle(frame, (40, 60), (w - 40, h - 60), (200, 205, 215), -1)
        cx, cy = w // 2, h // 2
        pitch_off = 0.12 if ("down" in label or "drowsy" in label) else 0.0
        head_y = cy + int(pitch_off * h)
        cv2.ellipse(frame, (cx, head_y), (70, 90), 0, 0, 360, (180, 150, 120), -1)
        # 眨眼：正常/alert 每 90 帧连续 5 帧闭眼（约 1.6 次/分钟）
        in_blink = (not fatigue) and (i % 90) < 5
        eye_open = (not fatigue or (i % 7) >= 4) and not in_blink
        eye_color = (0, 0, 0) if eye_open else (255, 255, 255)
        cv2.circle(frame, (cx - 25, head_y - 20), 8, eye_color, -1)
        cv2.circle(frame, (cx + 25, head_y - 20), 8, eye_color, -1)
        cv2.putText(frame, f"{video_id} [{label}]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()
    logger.info("合成视频已生成: %s（%s 秒）", out_path, seconds)


def _sample_from_raw(raw_dir: Path, samples: list, out_dir: Path) -> None:
    """从原始目录抽样复制。

    Args:
        raw_dir: 原始视频目录。
        samples: 抽样规格。
        out_dir: 输出目录。
    """
    import shutil

    for dataset, video_id, label in samples:
        cands = list((raw_dir / dataset).glob("*.mp4")) + list((raw_dir / dataset).glob("*.avi"))
        if cands:
            shutil.copy(cands[0], out_dir / f"{video_id}.mp4")
            logger.info("已从原始数据抽样: %s", cands[0])
        else:
            _make_synthetic(video_id, label, DAISEE_SECONDS if dataset == "daisee" else UTA_SECONDS, OUT_FPS)


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="离线处理 PoC 视频抽样器")
    parser.add_argument("--out_dir", default="datasets/poc_input", help="抽样输出目录")
    parser.add_argument("--raw_dir", default="datasets/raw", help="原始数据集目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    for dataset, video_id, label in DAISEE_SAMPLES + UTA_SAMPLES:
        cands = list((raw_dir / dataset).glob("*.mp4")) + list((raw_dir / dataset).glob("*.avi"))
        if cands:
            _sample_from_raw(raw_dir, [(dataset, video_id, label)], out_dir)
        else:
            _make_synthetic(video_id, label, DAISEE_SECONDS if dataset == "daisee" else UTA_SECONDS, OUT_FPS)


if __name__ == "__main__":
    main()
