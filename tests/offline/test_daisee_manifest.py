"""DAiSEE manifest 与标签关联的轻量测试。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.check_daisee import (
    MANIFEST_FIELDS,
    associate_labels,
    build_manifest_rows,
    discover_videos,
    load_labels,
)


def _create_fixture(root: Path) -> None:
    """创建 5 个视频文件和 5 条标签记录的隔离 fixture。"""
    files = {
        "Train": ("110001", "1100011002.avi"),
        "Validation": ("400022", "4000221001.avi"),
        "Test": ("500044", "5000441001.avi"),
    }
    for split, (subject_id, filename) in files.items():
        directory = root / "DataSet" / split / subject_id / Path(filename).stem
        directory.mkdir(parents=True)
        (directory / filename).write_bytes(b"fixture")
    extra = root / "DataSet" / "Train" / "110001" / "1100011003"
    extra.mkdir(parents=True)
    (extra / "1100011003.avi").write_bytes(b"fixture")
    extra = root / "DataSet" / "Test" / "500044" / "5000441002"
    extra.mkdir(parents=True)
    (extra / "5000441002.mp4").write_bytes(b"fixture")

    label_dir = root / "Labels"
    label_dir.mkdir()
    (label_dir / "TrainLabels.csv").write_text(
        "ClipID,Boredom,Engagement,Confusion,Frustration \n"
        " 1100011002.avi ,0,2,0,0\n"
        "2100552061.avi,2,3,2,0\n",
        encoding="utf-8",
    )
    (label_dir / "ValidationLabels.csv").write_text(
        "ClipID,Boredom,Engagement,Confusion,Frustration \n"
        "4000221001.avi,1,1,0,0\n",
        encoding="utf-8",
    )
    (label_dir / "TestLabels.csv").write_text(
        "ClipID,Boredom,Engagement,Confusion,Frustration \n"
        "5000441001.avi,3,0,1,0\n"
        "5000441002.mp4,0,3,0,1\n",
        encoding="utf-8",
    )


def test_manifest_has_five_rows_and_keeps_unlabeled_video(tmp_path: Path) -> None:
    """5 个视频对应 5 条 manifest，孤立标签不生成虚构视频行。"""
    _create_fixture(tmp_path)
    records = discover_videos(tmp_path)
    labels_by_split = {}
    issues = []
    for split in ("Train", "Validation", "Test"):
        labels, label_issues = load_labels(tmp_path / "Labels" / f"{split}Labels.csv", split)
        labels_by_split[split] = labels
        issues.extend(label_issues)
    issues.extend(associate_labels(records, labels_by_split))
    rows = build_manifest_rows(records)

    assert len(rows) == 5
    assert tuple(rows[0]) == MANIFEST_FIELDS
    assert set(labels_by_split["Train"]) == {"1100011002.avi", "2100552061.avi"}
    assert sum(row["engagement"] != "" for row in rows) == 4
    assert sum(row["engagement"] == "" for row in rows) == 1
    assert Counter(issue["issue_type"] for issue in issues) == {
        "orphan_label": 1,
        "unlabeled_video": 1,
    }
    linked = {row["clip_id"]: row["engagement"] for row in rows if row["engagement"] != ""}
    assert linked["1100011002.avi"] == 2
    assert linked["4000221001.avi"] == 1
    assert linked["5000441001.avi"] == 0
    assert linked["5000441002.mp4"] == 3
