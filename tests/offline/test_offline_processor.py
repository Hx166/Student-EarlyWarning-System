"""tests/offline/test_offline_processor.py —— 离线视频处理链路测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.service.offline_video_processor import OfflineVideoProcessor
from src.models.features.definitions import FEATURE_VECTOR_COLUMNS


def _write_synthetic_video(path: Path, seconds: float = 4.0, fps: float = 20.0) -> None:
    """写入合成测试视频。

    Args:
        path: 输出路径。
        seconds: 时长。
        fps: 帧率。
    """
    import cv2

    h, w = 240, 320
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "无法创建测试视频"
    for _ in range(int(seconds * fps)):
        frame = np.full((h, w, 3), 128, dtype=np.uint8)
        cv2.ellipse(frame, (w // 2, h // 2), (30, 40), 0, 0, 360, (180, 150, 120), -1)
        writer.write(frame)
    writer.release()


class OfflineVideoProcessorTest(unittest.TestCase):
    """离线处理器单元测试。"""

    def test_read_video_info(self) -> None:
        """读取视频基本参数。"""
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "test.mp4"
            _write_synthetic_video(video, seconds=2.0, fps=20.0)
            proc = OfflineVideoProcessor(output_dir=tmp)
            info = proc.read_video_info(str(video))
            self.assertGreaterEqual(info["fps"], 19.0)
            self.assertEqual(info["width"], 320)
            self.assertEqual(info["height"], 240)

    def test_process_video_creates_state_and_features(self) -> None:
        """处理视频应生成 StateRecord 与 11 维特征 CSV。"""
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "synthetic.mp4"
            _write_synthetic_video(video, seconds=4.0, fps=20.0)
            proc = OfflineVideoProcessor(output_dir=tmp, window_seconds=30)
            stats = proc.process_video(
                video_path=str(video),
                video_id="synth001",
                source_dataset="daisee",
                label="normal",
                subject_id="s001",
            )
            self.assertGreaterEqual(stats.state_record_count, 1)
            features_path = Path(tmp) / "features" / "synth001_features.csv"
            self.assertTrue(features_path.exists())
            import pandas as pd
            df = pd.read_csv(features_path)
            for col in FEATURE_VECTOR_COLUMNS:
                self.assertIn(col, df.columns)
            # 元数据保留且不作为特征
            self.assertIn("label", df.columns)
            self.assertIn("source_dataset", df.columns)
            self.assertIn("subject_id", df.columns)
            self.assertIn("video_id", df.columns)

    def test_features_metadata_not_in_feature_columns(self) -> None:
        """元数据列（source_dataset/subject_id/video_id）不得混入特征列。"""
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "meta.mp4"
            _write_synthetic_video(video, seconds=3.0, fps=15.0)
            proc = OfflineVideoProcessor(output_dir=tmp)
            proc.process_video(
                video_path=str(video),
                video_id="meta001",
                source_dataset="uta_rldd",
                label="drowsy",
                subject_id="s009",
            )
            import pandas as pd
            df = pd.read_csv(Path(tmp) / "features" / "meta001_features.csv")
            feature_subset = df[FEATURE_VECTOR_COLUMNS]
            self.assertNotIn("source_dataset", feature_subset.columns)
            self.assertNotIn("subject_id", feature_subset.columns)
            self.assertNotIn("video_id", feature_subset.columns)

    def test_state_records_match_video_seconds(self) -> None:
        """10 秒视频应生成约 10 条 StateRecord（秒级聚合不翻倍）。"""
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "second.mp4"
            _write_synthetic_video(video, seconds=4.0, fps=20.0)
            proc = OfflineVideoProcessor(output_dir=tmp)
            stats = proc.process_video(
                video_path=str(video),
                video_id="second001",
                source_dataset="daisee",
                label="normal",
                subject_id="s002",
            )
            self.assertEqual(stats.state_record_count, 4)


if __name__ == "__main__":
    unittest.main()
