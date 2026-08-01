"""src/api/routes/metrics.py —— 指标查询路由。

提供参与度、疲劳度等统计指标的查询接口。
"""

from __future__ import annotations

from flask import Blueprint, jsonify

from src.utils.logger import get_logger

logger = get_logger(__name__)

bp = Blueprint("metrics", __name__, url_prefix="/api/metrics")


@bp.route("/summary")
def summary():
    """获取课堂指标汇总。

    Returns:
        汇总数据 JSON 列表。
    """
    from src.api.app import db_service

    if db_service is None or db_service.conn is None:
        return jsonify({"code": 1, "data": [], "message": "数据库未连接"})

    records = db_service.query_state_records(limit=100)
    payload = {"total": len(records), "latest": records[0] if records else None}
    return jsonify({"code": 0, "data": payload, "message": "ok"})
