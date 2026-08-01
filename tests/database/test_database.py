"""tests/database/test_database.py —— 数据库模块测试。"""

from __future__ import annotations

import unittest

from src.database import DatabaseService


class DatabaseServiceTest(unittest.TestCase):
    """数据库服务的创建与读写测试。"""

    def setUp(self):
        """创建内存数据库连接。"""
        self.db = DatabaseService(db_path=":memory:").connect()

    def tearDown(self):
        """关闭数据库连接。"""
        self.db.close()

    def test_upsert_student_creates(self):
        """插入学生返回稳定 ID。"""
        sid = self.db.upsert_student("S0001", "张三", class_name="计科2301")
        self.assertGreater(sid, 0)

    def test_create_session(self):
        """创建会话返回正数 ID。"""
        sid = self.db.create_session("高数", "李老师", "A101")
        self.assertGreater(sid, 0)

    def test_insert_and_query_state(self):
        """写入并查询状态记录。"""
        student_id = self.db.upsert_student("S0002", "李四")
        session_id = self.db.create_session("英语", "王老师", "B201")
        self.db.insert_state_record(
            student_id, session_id, 0.3, 0.2, 5.0, 10.0, 2.0,
            0.1, 80.0, 20.0,
        )
        records = self.db.query_state_records(student_id=student_id)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["attention_score"], 80.0)

    def test_config_roundtrip(self):
        """配置键值写入与读取一致。"""
        self.db.set_config("threshold", "0.2")
        self.assertEqual(self.db.get_config("threshold"), "0.2")


if __name__ == "__main__":
    unittest.main()
