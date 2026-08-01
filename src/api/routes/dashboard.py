"""src/api/routes/dashboard.py —— Dashboard 页面与数据路由。

提供前端页面访问及状态数据查询接口。
"""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from src.utils.logger import get_logger

logger = get_logger(__name__)

bp = Blueprint("dashboard", __name__)


@bp.route("/")
def index():
    """渲染 Dashboard 首页。

    Returns:
        Dashboard 首页 HTML。
    """
    return render_template("index.html")


@bp.route("/api/health")
def health():
    """健康检查接口。

    Returns:
        服务健康状态 JSON。
    """
    return jsonify({"status": "ok", "service": "Student-EarlyWarning-System"})


@bp.route("/api/state")
def state():
    """查询最新课堂状态数据。

    Returns:
        状态数据 JSON 列表。
    """
    from src.api.app import db_service

    if db_service is None or db_service.conn is None:
        return jsonify({"code": 1, "data": [], "message": "数据库未连接"})

    limit = request.args.get("limit", 50, type=int)
    records = db_service.query_state_records(limit=limit)
    return jsonify({"code": 0, "data": records, "message": "ok"})
