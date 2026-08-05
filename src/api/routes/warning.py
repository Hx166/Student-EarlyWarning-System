"""src/api/routes/warning.py —— 预警、行为与帧数据查询路由。

供前端 Dashboard 实时展示风险预警、行为事件与原始帧数据。
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from src.utils.logger import get_logger

logger = get_logger(__name__)

bp = Blueprint("warning", __name__, url_prefix="/api")


@bp.route("/warnings")
def warnings():
    """\u67e5\u8be2\u65b0\u8fd1\u98ce\u9669\u9884\u8b66\u8bb0\u5f55\u3002

    Returns:
        \u9884\u8b66\u8bb0\u5f55 JSON \u5217\u8868\u3002
    """
    from src.api.app import db_service

    if db_service is None or db_service.conn is None:
        return jsonify({"code": 1, "data": [], "message": "\u6570\u636e\u5e93\u672a\u8fde\u63a5"})

    limit = request.args.get("limit", 50, type=int)
    student_id = request.args.get("student_id", type=int)
    records = db_service.query_warning_records(
        student_id=student_id, limit=limit
    )
    return jsonify({"code": 0, "data": records, "message": "ok"})


@bp.route("/behaviors")
def behaviors():
    """\u67e5\u8be2\u65b0\u8fd1\u884c\u4e3a\u4e8b\u4ef6\u8bb0\u5f55\u3002

    Returns:
        \u884c\u4e3a\u8bb0\u5f55 JSON \u5217\u8868\u3002
    """
    from src.api.app import db_service

    if db_service is None or db_service.conn is None:
        return jsonify({"code": 1, "data": [], "message": "\u6570\u636e\u5e93\u672a\u8fde\u63a5"})

    limit = request.args.get("limit", 50, type=int)
    records = db_service.query_behavior_records(limit=limit)
    return jsonify({"code": 0, "data": records, "message": "ok"})


@bp.route("/frames")
def frames():
    """\u67e5\u8be2\u65b0\u8fd1\u539f\u59cb\u5e27\u89c6\u89c9\u6570\u636e\u3002

    Returns:
        \u5e27\u8bb0\u5f55 JSON \u5217\u8868\u3002
    """
    from src.api.app import db_service

    if db_service is None or db_service.conn is None:
        return jsonify({"code": 1, "data": [], "message": "\u6570\u636e\u5e93\u672a\u8fde\u63a5"})

    limit = request.args.get("limit", 50, type=int)
    records = db_service.query_frame_records(limit=limit)
    return jsonify({"code": 0, "data": records, "message": "ok"})
