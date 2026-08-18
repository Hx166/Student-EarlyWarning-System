"""tests/database/test_prediction.py —— 预测记录表测试。"""

from __future__ import annotations

import unittest

from src.database import DatabaseService


class PredictionLogTest(unittest.TestCase):
    """PredictionLog 写入与查询测试。"""

    def setUp(self):
        """创建内存数据库连接。"""
        self.db = DatabaseService(db_path=":memory:").connect()
        self.student_id = self.db.upsert_student("S1001", "预测学生")
        self.session_id = self.db.create_session("高数", "李老师", "A101")

    def tearDown(self):
        """关闭数据库连接。"""
        self.db.close()

    def test_insert_and_query_prediction(self):
        """写入并查询预测记录。"""
        pid = self.db.insert_prediction(
            student_id=self.student_id,
            session_id=self.session_id,
            risk_label="fatigue",
            risk_probability=0.72,
            risk_level="warning",
            model_version="v1",
        )
        self.assertGreater(pid, 0)

        records = self.db.query_prediction_records(student_id=self.student_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["risk_label"], "fatigue")
        self.assertAlmostEqual(records[0]["risk_probability"], 0.72)
        self.assertEqual(records[0]["risk_level"], "warning")
        self.assertEqual(records[0]["model_version"], "v1")

    def test_query_prediction_feature_json(self):
        """feature_json 可选字段应正确存储。"""
        self.db.insert_prediction(
            student_id=self.student_id,
            session_id=self.session_id,
            risk_label="high_risk",
            risk_probability=0.9,
            risk_level="high_risk",
            model_version="v1",
            feature_json='{"ear_mean": 0.2}',
        )
        records = self.db.query_prediction_records()
        self.assertEqual(records[0]["feature_json"], '{"ear_mean": 0.2}')

    def test_query_prediction_empty(self):
        """无记录时返回空列表。"""
        self.assertEqual(self.db.query_prediction_records(), [])


if __name__ == "__main__":
    unittest.main()
