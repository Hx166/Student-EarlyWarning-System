"""src/api/routes/predict.py —— 预测相关路由。

提供在线预测、历史预测查询与当前模型信息接口。
"""

from __future__ import annotations

import pandas as pd
from flask import Blueprint, jsonify, request

from src.api.service.prediction import (
    get_model_registry,
    get_model_name,
    get_model_version,
    get_risk_predictor,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

bp = Blueprint("predict", __name__, url_prefix="/api")


@bp.route("/predict", methods=["POST"])
def predict():
    """在线风险预测接口。

    请求体：{"features": { ...11 列特征向量... }}

    Returns:
        风险预测结果 JSON（复用 RiskPredictionResult 结构）。
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "features" not in payload:
        return jsonify({"code": 1, "data": None, "message": "缺少 features 字段"}), 400

    features = payload["features"]
    if not isinstance(features, dict):
        return jsonify({"code": 1, "data": None, "message": "features 必须为特征向量对象"}), 400

    try:
        df = pd.DataFrame([features])
    except (ValueError, TypeError) as exc:
        return jsonify({"code": 1, "data": None, "message": f"特征格式错误: {exc}"}), 400

    try:
        result = get_risk_predictor().predict(df)
    except ValueError as exc:
        return jsonify({"code": 1, "data": None, "message": f"特征校验失败: {exc}"}), 400
    except RuntimeError as exc:
        return jsonify({"code": 1, "data": None, "message": f"模型不可用: {exc}"}), 503

    return jsonify({"code": 0, "data": result.to_dict(), "message": "ok"})


@bp.route("/predictions")
def predictions():
    """查询历史预测记录。

    Returns:
        预测记录 JSON 列表。
    """
    from src.api.app import db_service

    if db_service is None or db_service.conn is None:
        return jsonify({"code": 1, "data": [], "message": "数据库未连接"})

    limit = request.args.get("limit", 50, type=int)
    student_id = request.args.get("student_id", type=int)
    records = db_service.query_prediction_records(
        student_id=student_id, limit=limit
    )
    return jsonify({"code": 0, "data": records, "message": "ok"})


@bp.route("/model/info")
def model_info():
    """查询当前加载模型信息。

    Returns:
        模型版本、特征名与元数据状态。
    """
    registry = get_model_registry()
    version = get_model_version()
    status = registry.check_model(get_model_name(), version=version)
    return jsonify({"code": 0, "data": status, "message": "ok"})
