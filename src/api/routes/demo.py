"""阶段 G：实时看板、视频回放与模型说明页面路由。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint, jsonify, render_template, send_file

from src.service.demo_playback import DemoPlaybackService
from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)
bp = Blueprint("demo", __name__)
_demo_service: Optional[DemoPlaybackService] = None


def get_demo_service() -> DemoPlaybackService:
    """返回演示回放服务单例。"""
    global _demo_service
    if _demo_service is None:
        _demo_service = DemoPlaybackService()
    return _demo_service


def reset_demo_service() -> None:
    """清理演示服务单例（测试用）。"""
    global _demo_service
    _demo_service = None


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON；缺失或损坏时返回空字典。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


@bp.route("/playback")
def playback_page():
    """渲染视频回放页面。"""
    return render_template("playback.html")


@bp.route("/model")
def model_page():
    """渲染模型说明页面。"""
    root = Path(PROJECT_ROOT)
    metrics = _read_json(root / "outputs" / "stage_e" / "metrics.json")
    metadata = _read_json(
        root / "models" / "xgboost" / "engagement-v1" / "model_metadata.json"
    )
    payload = {
        "available": bool(metrics and metadata),
        "metrics": metrics,
        "metadata": metadata,
    }
    return render_template("model.html", model_data=payload)


@bp.route("/api/demo/clips")
def demo_clips():
    """返回冻结清单中的回放片段。"""
    try:
        data = get_demo_service().get_clips()
    except ValueError as exc:
        logger.error("读取演示片段失败: %s", exc)
        return jsonify({"code": 1, "data": [], "message": "演示数据不可用"}), 500
    return jsonify({"code": 0, "data": data, "message": "ok"})


@bp.route("/api/demo/run/<clip_id>")
def demo_run(clip_id: str):
    """处理指定片段并返回曲线、标签与概率。"""
    try:
        data = get_demo_service().run(clip_id)
    except KeyError:
        return jsonify({"code": 1, "data": None, "message": "片段不存在"}), 404
    except FileNotFoundError:
        return jsonify({"code": 1, "data": None, "message": "视频文件不存在"}), 404
    except Exception as exc:
        logger.exception("演示回放失败: %s", exc)
        return jsonify({"code": 1, "data": None, "message": "演示数据不可用"}), 500
    return jsonify({"code": 0, "data": data, "message": "ok"})


@bp.route("/api/demo/video/<clip_id>")
def demo_video(clip_id: str):
    """返回清单内 MP4 文件。"""
    try:
        path = get_demo_service().video_path(clip_id)
    except (KeyError, FileNotFoundError):
        return jsonify({"code": 1, "data": None, "message": "视频不存在"}), 404
    return send_file(path, mimetype="video/mp4", conditional=True)
