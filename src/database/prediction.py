"""src/database/prediction.py —— 预测记录数据访问（Mixin）。

为 DatabaseService 提供 PredictionLog 表的写入与查询能力，
通过 Mixin 组合避免 service.py 单文件过长（遵守单文件 500 行限制）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


class PredictionMixin:
    """PredictionLog 表访问方法混入。

    依赖宿主类（DatabaseService）提供的 conn 属性。
    """

    def insert_prediction(
        self,
        student_id: int,
        session_id: int,
        risk_label: str,
        risk_probability: float,
        risk_level: str,
        model_version: str,
        feature_json: Optional[str] = None,
    ) -> int:
        """插入模型预测记录。

        Args:
            student_id: 学生 ID。
            session_id: 会话 ID。
            risk_label: 风险类别标签。
            risk_probability: 风险概率。
            risk_level: 风险等级。
            model_version: 使用的模型版本。
            feature_json: 可选特征快照 JSON 字符串。

        Returns:
            记录 ID。
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO PredictionLog(
                student_id, session_id, record_time, risk_label,
                risk_probability, risk_level, model_version, feature_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id, session_id, datetime.now().isoformat(),
                risk_label, risk_probability, risk_level, model_version,
                feature_json,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def query_prediction_records(
        self, student_id: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查询模型预测记录。

        Args:
            student_id: 学生 ID 过滤条件，None 表示所有学生。
            limit: 返回记录条数上限。

        Returns:
            预测记录字典列表。
        """
        query = "SELECT * FROM PredictionLog"
        params: List[Any] = []
        if student_id is not None:
            query += " WHERE student_id = ?"
            params.append(student_id)
        query += " ORDER BY record_time DESC LIMIT ?"
        params.append(limit)
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
