"""阶段 D 合并、校验和断点续跑测试，不依赖 cv2/mediapipe。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_daisee_features import (
    FEATURE_CSV_COLUMNS,
    FEATURE_VECTOR_COLUMNS,
    atomic_write_json,
    merge_and_validate,
    pending_records,
    validation_errors,
)


def _row(clip_id: str, split: str, engagement: int, index: int) -> dict[str, object]:
    """生成一条完整合成特征行。"""
    row: dict[str, object] = {
        "clip_id": clip_id,
        "subject_id": f"S{index}",
        "split": split,
        "engagement": engagement,
        "aggregation": "A",
        "frames": 300,
        "seconds": 10.0,
        "face_found_ratio": 1.0,
        "elapsed_seconds": 1.0,
    }
    row.update({feature: float(index + offset) for offset, feature in enumerate(FEATURE_VECTOR_COLUMNS)})
    return row


def test_merge_validates_five_rows_and_resume_skips_valid_json(tmp_path: Path) -> None:
    """5 条行可合并校验，合法断点行会被跳过。"""
    rows = [
        _row("a", "Train", 0, 1),
        _row("b", "Train", 1, 2),
        _row("c", "Train", 3, 3),
        _row("d", "Validation", 2, 4),
        _row("e", "Test", 3, 5),
    ]
    records = [
        {
            "clip_id": row["clip_id"],
            "subject_id": row["subject_id"],
            "split": row["split"],
            "engagement": row["engagement"],
            "frames": row["frames"],
            "path": str(tmp_path / str(row["clip_id"])),
        }
        for row in rows
    ]
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    rows_dir = tmp_path / "rows"
    for row in rows:
        atomic_write_json(rows_dir / f"{row['clip_id']}.json", row)

    output_path = tmp_path / "features.csv"
    stats_path = tmp_path / "stats.csv"
    frame, stats, maximum = merge_and_validate(
        rows_dir,
        manifest_path,
        output_path,
        stats_path,
        expected_rows=5,
        expected_splits={"Train": 3, "Validation": 1, "Test": 1},
        expected_levels={0: 1, 1: 1, 2: 1, 3: 2},
    )

    assert len(frame) == 5
    assert frame["clip_id"].is_unique
    assert list(frame.columns) == list(FEATURE_CSV_COLUMNS)
    assert frame.groupby("split").size().to_dict() == {"Test": 1, "Train": 3, "Validation": 1}
    assert frame.groupby("engagement").size().to_dict() == {0: 1, 1: 1, 2: 1, 3: 2}
    assert len(stats) == 11
    assert maximum == 0.0
    assert output_path.exists() and stats_path.exists()
    assert [item["clip_id"] for item in pending_records(records, rows_dir)] == []

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    errors = validation_errors(
        duplicate,
        pd.read_csv(manifest_path),
        expected_rows=5,
        expected_splits={"Train": 3, "Validation": 1, "Test": 1},
        expected_levels={0: 1, 1: 1, 2: 1, 3: 2},
    )
    assert any("clip_id 存在重复" in error for error in errors)

    manifest = pd.read_csv(manifest_path)
    lower_frames = frame.copy()
    lower_frames.loc[0, "frames"] = 296
    assert not validation_errors(
        lower_frames,
        manifest,
        expected_rows=5,
        expected_splits={"Train": 3, "Validation": 1, "Test": 1},
        expected_levels={0: 1, 1: 1, 2: 1, 3: 2},
    )
    higher_frames = frame.copy()
    higher_frames.loc[0, "frames"] = 301
    assert any(
        "实际解码帧数大于 manifest 帧数" in error
        for error in validation_errors(
            higher_frames,
            manifest,
            expected_rows=5,
            expected_splits={"Train": 3, "Validation": 1, "Test": 1},
            expected_levels={0: 1, 1: 1, 2: 1, 3: 2},
        )
    )


def test_pending_records_retries_missing_and_corrupt_json(tmp_path: Path) -> None:
    """缺失或损坏断点行返回待处理，合法 JSON 被跳过。"""
    records = [
        {"clip_id": "done"},
        {"clip_id": "corrupt"},
        {"clip_id": "missing"},
    ]
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    atomic_write_json(rows_dir / "done.json", {"clip_id": "done"})
    (rows_dir / "corrupt.json").write_text("{not-json", encoding="utf-8")
    pending = pending_records(records, rows_dir)
    assert [item["clip_id"] for item in pending] == ["corrupt", "missing"]
