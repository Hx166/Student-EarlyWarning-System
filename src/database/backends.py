"""src/database/backends.py —— 数据库存储后端抽象。

将 DatabaseService 的底层连接与执行细节抽象为可插拔后端：
- SQLiteBackend：默认后端，保持既有行为不变；
- MySQLBackend：新增支持，通过配置切换（惰性导入 pymysql）。

后端统一提供 execute / query / query_one 接口，
对占位符差异（SQLite ? 与 MySQL %s）与行转换做内部处理，
保证 DatabaseService 上层 SQL 无需感知方言差异。
"""

from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.database.schema import create_indexes, create_tables
from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_sqlite_path(url: str) -> str:
    """解析 sqlite url 为文件路径。

    Args:
        url: sqlite 连接字符串，如 "sqlite:///path" 或 ":memory:"。

    Returns:
        解析后的数据库路径。
    """
    if url == ":memory:" or url == "sqlite:///:memory:":
        return ":memory:"
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
    else:
        path = url
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    return path


# MySQL 建表语句（字段名与 SQLite 一致，仅类型/约束适配 MySQL 方言）
MYSQL_DDL_STATEMENTS: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS Student(
        student_id INT PRIMARY KEY AUTO_INCREMENT,
        student_no VARCHAR(64) UNIQUE,
        student_name VARCHAR(128),
        gender VARCHAR(16),
        class_name VARCHAR(128),
        major VARCHAR(128),
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS CourseSession(
        session_id INT PRIMARY KEY AUTO_INCREMENT,
        course_name VARCHAR(128),
        teacher_name VARCHAR(128),
        classroom VARCHAR(128),
        start_time DATETIME,
        end_time DATETIME
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS FrameRecord(
        frame_id INT PRIMARY KEY AUTO_INCREMENT,
        student_id INT,
        session_id INT,
        frame_time DATETIME,
        ear DOUBLE,
        mar DOUBLE,
        pitch DOUBLE,
        yaw DOUBLE,
        roll DOUBLE,
        FOREIGN KEY(student_id) REFERENCES Student(student_id),
        FOREIGN KEY(session_id) REFERENCES CourseSession(session_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS StateRecord(
        state_id INT PRIMARY KEY AUTO_INCREMENT,
        student_id INT,
        session_id INT,
        record_time DATETIME,
        avg_ear DOUBLE,
        avg_mar DOUBLE,
        avg_pitch DOUBLE,
        avg_yaw DOUBLE,
        avg_roll DOUBLE,
        perclos DOUBLE,
        attention_score DOUBLE,
        fatigue_score DOUBLE,
        FOREIGN KEY(student_id) REFERENCES Student(student_id),
        FOREIGN KEY(session_id) REFERENCES CourseSession(session_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS BehaviorRecord(
        behavior_id INT PRIMARY KEY AUTO_INCREMENT,
        student_id INT,
        session_id INT,
        behavior_type VARCHAR(32),
        start_time DATETIME,
        end_time DATETIME,
        duration DOUBLE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS WarningLog(
        warning_id INT PRIMARY KEY AUTO_INCREMENT,
        student_id INT,
        session_id INT,
        warning_level VARCHAR(16),
        warning_type VARCHAR(32),
        warning_reason TEXT,
        attention_score DOUBLE,
        create_time DATETIME,
        processed INT DEFAULT 0
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS SystemConfig(
        id INT PRIMARY KEY AUTO_INCREMENT,
        config_key VARCHAR(128),
        config_value TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS PredictionLog(
        prediction_id INT PRIMARY KEY AUTO_INCREMENT,
        student_id INT,
        session_id INT,
        record_time DATETIME,
        risk_label VARCHAR(32),
        risk_probability DOUBLE,
        risk_level VARCHAR(16),
        model_version VARCHAR(32),
        feature_json TEXT,
        FOREIGN KEY(student_id) REFERENCES Student(student_id),
        FOREIGN KEY(session_id) REFERENCES CourseSession(session_id)
    );
    """,
]

MYSQL_INDEX_STATEMENTS: List[str] = [
    "CREATE INDEX idx_student_no ON Student(student_no);",
    "CREATE INDEX idx_state_student ON StateRecord(student_id);",
    "CREATE INDEX idx_state_time ON StateRecord(record_time);",
    "CREATE INDEX idx_warning_student ON WarningLog(student_id);",
    "CREATE INDEX idx_frame_student ON FrameRecord(student_id);",
    "CREATE INDEX idx_frame_time ON FrameRecord(frame_time);",
    "CREATE INDEX idx_prediction_student ON PredictionLog(student_id);",
    "CREATE INDEX idx_prediction_time ON PredictionLog(record_time);",
]


class DatabaseBackend(ABC):
    """数据库后端抽象基类。

    Attributes:
        placeholder: SQL 占位符（SQLite 为 ?，MySQL 为 %s）。
    """

    placeholder: str = "?"

    @abstractmethod
    def connect(self) -> Any:
        """建立连接。"""

    @abstractmethod
    def create_schema(self, conn: Any) -> None:
        """创建表结构与索引。"""

    @abstractmethod
    def close(self, conn: Any) -> None:
        """关闭连接。"""

    def _translate(self, sql: str) -> str:
        """按方言翻译占位符。

        Args:
            sql: 使用 ? 占位符的 SQL。

        Returns:
            翻译后的 SQL。
        """
        if self.placeholder == "?":
            return sql
        return sql.replace("?", self.placeholder)

    def execute(self, conn: Any, sql: str, params: Tuple = ()) -> int:
        """执行写语句并返回 lastrowid。

        Args:
            conn: 连接对象。
            sql: SQL 语句（? 占位符）。
            params: 参数元组。

        Returns:
            自增主键 id（无则返回 None）。
        """
        cursor = conn.cursor()
        cursor.execute(self._translate(sql), params)
        conn.commit()
        return getattr(cursor, "lastrowid", None)

    def query(self, conn: Any, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """执行查询并返回字典列表。

        Args:
            conn: 连接对象。
            sql: SQL 语句（? 占位符）。
            params: 参数元组。

        Returns:
            行字典列表。
        """
        cursor = conn.cursor()
        cursor.execute(self._translate(sql), params)
        return [dict(row) for row in cursor.fetchall()]

    def query_one(
        self, conn: Any, sql: str, params: Tuple = ()
    ) -> Optional[Dict[str, Any]]:
        """执行查询并返回单行字典。

        Args:
            conn: 连接对象。
            sql: SQL 语句（? 占位符）。
            params: 参数元组。

        Returns:
            行字典；无结果返回 None。
        """
        cursor = conn.cursor()
        cursor.execute(self._translate(sql), params)
        row = cursor.fetchone()
        return dict(row) if row is not None else None


class SQLiteBackend(DatabaseBackend):
    """SQLite 后端（默认）。

    Attributes:
        db_path: 数据库文件路径。
        foreign_keys: 是否启用外键约束。
    """

    placeholder = "?"

    def __init__(self, db_path: str, foreign_keys: bool = True) -> None:
        """初始化 SQLite 后端。

        Args:
            db_path: 数据库文件路径（含 :memory:）。
            foreign_keys: 是否启用外键约束。
        """
        self.db_path = db_path
        self.foreign_keys = foreign_keys

    def connect(self) -> sqlite3.Connection:
        """建立 SQLite 连接。

        Returns:
            SQLite 连接对象。

        Raises:
            sqlite3.Error: 连接失败时抛出。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if self.foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def create_schema(self, conn: sqlite3.Connection) -> None:
        """创建 SQLite 表与索引。

        Args:
            conn: SQLite 连接对象。
        """
        create_tables(conn)
        create_indexes(conn)

    def close(self, conn: sqlite3.Connection) -> None:
        """关闭 SQLite 连接。

        Args:
            conn: SQLite 连接对象。
        """
        conn.close()


class MySQLBackend(DatabaseBackend):
    """MySQL 后端（可选，需安装 pymysql）。

    Attributes:
        host: 主机。
        port: 端口。
        user: 用户名。
        password: 密码。
        database: 数据库名。
        charset: 字符集。
    """

    placeholder = "%s"

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """初始化 MySQL 后端。

        Args:
            cfg: MySQL 配置字典（host/port/user/password/database/charset）。
        """
        self.host = cfg.get("host", "127.0.0.1")
        self.port = int(cfg.get("port", 3306))
        self.user = cfg.get("user", "root")
        self.password = cfg.get("password", "")
        self.database = cfg.get("database", "early_warning")
        self.charset = cfg.get("charset", "utf8mb4")

    def connect(self) -> Any:
        """建立 MySQL 连接。

        Returns:
            pymysql 连接对象。

        Raises:
            ImportError: pymysql 未安装时抛出。
        """
        try:
            import pymysql
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "未安装 pymysql，无法使用 MySQL 后端，请执行 pip install pymysql"
            ) from exc
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def create_schema(self, conn: Any) -> None:
        """创建 MySQL 表与索引。

        Args:
            conn: MySQL 连接对象。
        """
        cursor = conn.cursor()
        for statement in MYSQL_DDL_STATEMENTS:
            cursor.execute(statement)
        for statement in MYSQL_INDEX_STATEMENTS:
            try:
                cursor.execute(statement)
            except Exception as exc:  # 重复索引等可忽略错误
                logger.warning("创建索引跳过（%s）: %s", statement[:40], exc)
        conn.commit()

    def close(self, conn: Any) -> None:
        """关闭 MySQL 连接。

        Args:
            conn: MySQL 连接对象。
        """
        conn.close()


def create_backend(
    name: str, cfg: Dict[str, Any], db_path: Optional[str] = None
) -> DatabaseBackend:
    """根据配置创建数据库后端。

    Args:
        name: 后端名称（sqlite / mysql）。
        cfg: 完整数据库配置字典。
        db_path: SQLite 数据库路径（显式指定，测试用）。

    Returns:
        后端实例。

    Raises:
        ValueError: 不支持的后端名称。
    """
    db_cfg = cfg.get("database", {})
    if name == "mysql":
        return MySQLBackend(cfg.get("mysql", {}))
    if name in ("", "sqlite", None):
        if db_path is None:
            raw = db_cfg.get("url", "sqlite:///./data/early_warning.db")
            db_path = _resolve_sqlite_path(raw)
        return SQLiteBackend(db_path, foreign_keys=bool(db_cfg.get("foreign_keys", True)))
    raise ValueError(f"不支持的数据库后端: {name}")
