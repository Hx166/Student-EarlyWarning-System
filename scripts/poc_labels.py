"""scripts/poc_labels.py —— 为 PoC 生成标签映射文件。

按可行性调查的语义映射生成 datasets/labels/*.json：
- DAiSEE：Engagement 分箱 → normal / attention_down
- UTA-RLDD：alert / low_vigilant / drowsy → normal / attention_down / fatigue

不生成 high_risk 标签，也不修改生产代码。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# DAiSEE 官方 4 级 Engagement → 训练标签（仅参与度语义）
DAISEE_ENGAGEMENT_TO_LABEL = {
    0: "attention_down",
    1: "attention_down",
    2: "normal",
    3: "normal",
}

# UTA-RLDD 原始标签 → 训练标签
UTA_RAW_TO_LABEL = {
    "alert": "normal",
    "low_vigilant": "attention_down",
    "drowsy": "fatigue",
}


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="生成 PoC 标签映射")
    parser.add_argument("--output_dir", default="datasets/labels")
    parser.add_argument("--daisee_videos", nargs="*", default=["daisee_s001_001", "daisee_s002_002"])
    parser.add_argument("--daisee_engagement", nargs="*", type=int, default=[3, 1])
    parser.add_argument("--uta_videos", nargs="*", default=["uta_s001_alert", "uta_s002_lowvigilant", "uta_s003_drowsy"])
    parser.add_argument("--uta_labels", nargs="*", default=["alert", "low_vigilant", "drowsy"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daisee_map = {}
    for vid, eng in zip(args.daisee_videos, args.daisee_engagement):
        daisee_map[vid] = DAISEE_ENGAGEMENT_TO_LABEL[int(eng)]
    (out_dir / "daisee_labels.json").write_text(
        json.dumps(daisee_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    uta_map = {}
    for vid, lab in zip(args.uta_videos, args.uta_labels):
        uta_map[vid] = UTA_RAW_TO_LABEL.get(lab, "normal")
    (out_dir / "uta_rldd_labels.json").write_text(
        json.dumps(uta_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DAiSEE:", daisee_map)
    print("UTA:", uta_map)


if __name__ == "__main__":
    main()
