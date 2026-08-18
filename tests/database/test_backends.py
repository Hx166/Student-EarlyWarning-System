"""tests/database/test_backends.py —— 数据库后端抽象测试。"""

from __future__ import annotations

import unittest

from src.database import DatabaseService
from src.database.backends import (
    MYSQL_DDL_STATEMENTS,
    MYSQL_INDEX_STATEMENTS,
    MySQLBackend,
    SQLiteBackend,
    create_backend,
)


class FakeCursor:
    """模拟数据库游标。"""

    def __init__(self, rows=None):
        self.rows = rows or [{"a": 1, "b": 2}]
        self.executed_sql = None
        self.executed_params = None
        self.lastrowid = 7

    def execute(self, sql, params=()):
        self.executed_sql = sql
        self.executed_params = params

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    """模拟数据库连接。"""

    def __init__(self):
        self._cursor = FakeCursor()
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        pass


class BackendSelectionTest(unittest.TestCase):
    """后端创建与选择测试。"""

    def test_create_backend_sqlite(self):
        """默认应返回 SQLiteBackend。"""
        backend = create_backend("sqlite", {"database": {"url": "sqlite:///:memory:"}})
        self.assertIsInstance(backend, SQLiteBackend)

    def test_create_backend_mysql(self):
        """mysql 应返回 MySQLBackend。"""
        backend = create_backend("mysql", {"database": {}, "mysql": {"host": "h"}})
        self.assertIsInstance(backend, MySQLBackend)

    def test_create_backend_unknown_raises(self):
        """未知后端应抛 ValueError。"""
        with self.assertRaises(ValueError):
            create_backend("oracle", {"database": {}})

    def test_database_service_backend_selection(self):
        """DatabaseService 应根据 backend 参数选择后端。"""
        service = DatabaseService(backend="mysql")
        self.assertIsInstance(service._backend, MySQLBackend)


class MySQLBackendTest(unittest.TestCase):
    """MySQL 后端占位符翻译与行转换测试（不依赖 pymysql）。"""

    def setUp(self):
        self.backend = MySQLBackend({})

    def test_execute_translates_placeholder(self):
        """execute 应将 ? 翻译为 %s 并提交。"""
        conn = FakeConn()
        self.backend.execute(conn, "INSERT INTO x VALUES (?, ?)", (1, 2))
        self.assertEqual(conn._cursor.executed_sql, "INSERT INTO x VALUES (%s, %s)")
        self.assertEqual(conn._cursor.executed_params, (1, 2))
        self.assertTrue(conn.committed)

    def test_query_returns_dicts(self):
        """query 应返回字典列表并翻译占位符。"""
        conn = FakeConn()
        rows = self.backend.query(conn, "SELECT * FROM x WHERE a = ?", (1,))
        self.assertEqual(rows, [{"a": 1, "b": 2}])
        self.assertEqual(conn._cursor.executed_sql, "SELECT * FROM x WHERE a = %s")

    def test_query_one_returns_dict(self):
        """query_one 应返回单行字典。"""
        conn = FakeConn()
        row = self.backend.query_one(conn, "SELECT * FROM x WHERE a = ?", (1,))
        self.assertEqual(row, {"a": 1, "b": 2})

    def test_mysql_ddl_covers_tables(self):
        """MySQL DDL 应包含全部表。"""
        tables = [
            "Student", "CourseSession", "FrameRecord", "StateRecord",
            "BehaviorRecord", "WarningLog", "SystemConfig", "PredictionLog",
        ]
        ddl = " ".join(MYSQL_DDL_STATEMENTS)
        for table in tables:
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", ddl)
        self.assertTrue(len(MYSQL_INDEX_STATEMENTS) > 0)


class SQLiteBackendTest(unittest.TestCase):
    """SQLite 后端基础读写测试。"""

    def test_sqlite_backend_roundtrip(self):
        """SQLite 后端建表与读写应正常。"""
        backend = SQLiteBackend(":memory:")
        conn = backend.connect()
        backend.create_schema(conn)
        backend.execute(
            conn,
            "INSERT INTO SystemConfig(config_key, config_value) VALUES (?, ?)",
            ("k", "v"),
        )
        row = backend.query_one(
            conn,
            "SELECT config_value FROM SystemConfig WHERE config_key = ?",
            ("k",),
        )
        self.assertEqual(row["config_value"], "v")
        backend.close(conn)


if __name__ == "__main__":
    unittest.main()
