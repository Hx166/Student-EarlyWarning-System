"""tests/api/test_api.py —— API 模块测试。"""

from __future__ import annotations

import unittest

from src.api.app import create_app
from src.database import DatabaseService


class ApiTest(unittest.TestCase):
    """Flask API 测试。"""

    def setUp(self):
        """创建测试应用与内存数据库。"""
        self.db = DatabaseService(db_path=":memory:").connect()
        self.app = create_app(database=self.db)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # 插入一条状态数据
        sid = self.db.upsert_student("S0001", "张三")
        seid = self.db.create_session("高数", "李老师", "A101")
        self.db.insert_state_record(
            sid, seid, 0.3, 0.2, 5.0, 10.0, 2.0, 0.1, 82.0, 18.0,
        )

    def tearDown(self):
        """关闭数据库。"""
        self.db.close()

    def test_health(self):
        """健康检查接口返回 ok。"""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "ok")

    def test_state_record(self):
        """状态接口返回写入的数据。"""
        res = self.client.get("/api/state?limit=10")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["code"], 0)
        self.assertEqual(len(payload["data"]), 1)

    def test_index_page(self):
        """首页可访问。"""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
