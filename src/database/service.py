"""src/database/service.py —— 数据库统一访问服务。

提供 SQLite 连接的建立、表的初始化以及常用数据的写入与查询能力。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import get_database
from src.database.schema import create_indexes, create_tables
from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT

logger = get_logger(__name__)


class DatabaseService:
    """数据库访问服务。

    负责 SLQite 连接管理与数据持久化操作。

    Attributes:
        conn: SQLite 连接对象。
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """初始化数据库服务。

        Args:
            db_path: 数据库文件路径；为 None 时从配置读取。
        """
        if db_path is None:
            cfg = get_database()
            raw = cfg.get("database", {}).get("url", "sqlite:///./data/early_warning.db")
            db_path = self._resolve_sqlite_path(raw)
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def _resolve_sqlite_path(self, url: str) -> str:
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
        # 相对路径基于项目根目录解析
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        return path

    def connect(self) -> "DatabaseService":
        """建立数据库连接并初始化表结构。

        Returns:
            当前实例，便于链式调用。

        Raises:
            sqlite3.Error: 连接失败时抛出。
        """
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        cfg = get_database()
        if cfg.get("database", {}).get("foreign_keys", True):
            self.conn.execute("PRAGMA foreign_keys = ON;")
        create_tables(self.conn)
        create_indexes(self.conn)
        logger.info("数据库连接成功: %s", self.db_path)
        return self

    def close(self) -> None:
        """关闭数据库连接。"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("数据库连接已关闭")

    def upsert_student(
        self,
        student_no: str,
        student_name: str,
        gender: str = "",
        class_name: str = "",
        major: str = "",
    ) -> int:
        """插入或更新学生信息。

        Args:
            student_no: 学号（唯一）。
            student_name: 姓名。
            gender: 性别。
            class_name: 班级。
            major: 专业。

        Returns:
            学生 ID。
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT student_id FROM Student WHERE student_no = ?",
            (student_no,),
        )
        row = cursor.fetchone()
        if row is not None:
            cursor.execute(
                """
                UPDATE Student SET student_name=?, gender=?, class_name=?, major=?
                WHERE student_no=?
                """,
                (student_name, gender, class_name, major, student_no),
            )
            student_id = row["student_id"]
        else:
            cursor.execute(
                """
                INSERT INTO Student(student_no, student_name, gender, class_name, major)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_no, student_name, gender, class_name, major),
            )
            student_id = cursor.lastrowid
        self.conn.commit()
        return student_id

    def create_session(
        self,
        course_name: str,
        teacher_name: str,
        classroom: str,
        start_time: Optional[datetime] = None,
    ) -> int:
        """创建课堂会话。

        Args:
            course_name: 课程名称。
            teacher_name: 教师姓名。
            classroom: 教室。
            start_time: 开始时间，None 时用当前时间。

        Returns:
            会话 ID。
        """
        if start_time is None:
            start_time = datetime.now()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO CourseSession(course_name, teacher_name, classroom, start_time)
            VALUES (?, ?, ?, ?)
            """,
            (course_name, teacher_name, classroom, start_time.isoformat()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def end_session(self, session_id: int) -> None:
        """结束课堂会话。

        Args:
            session_id: 会话 ID。
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE CourseSession SET end_time=? WHERE session_id=?",
            (datetime.now().isoformat(), session_id),
        )
        self.conn.commit()

    def insert_frame_record(
        self,
        student_id: int,
        session_id: int,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> int:
        """插入原始帧视觉数据（科研用）。

        Args:
            student_id: 学生 ID。
            session_id: 会话 ID。
            ear: EAR 值。
            mar: MAR 值。
            pitch: Pitch 角。
            yaw: Yaw 角。
            roll: Roll 角。

        Returns:
            记录 ID。
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO FrameRecord(
                student_id, session_id, frame_time, ear, mar, pitch, yaw, roll
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id, session_id, datetime.now().isoformat(),
                ear, mar, pitch, yaw, roll,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_state_record(
        self,
        student_id: int,
        session_id: int,
        avg_ear: float,
        avg_mar: float,
        avg_pitch: float,
        avg_yaw: float,
        avg_roll: float,
        perclos: float,
        attention_score: float,
        fatigue_score: float,
    ) -> int:
        """插入每秒状态统计数据。

        Args:
            student_id: 学生 ID。
            session_id: 会话 ID。
            avg_ear: 平均 EAR。
            avg_mar: 平均 MAR。
            avg_pitch: 平均 Pitch。
            avg_yaw: 平均 Yaw。
            avg_roll: 平均 Roll。
            perclos: PERCLOS 值。
            attention_score: 专注度评分。
            fatigue_score: 疲劳评分。

        Returns:
            记录 ID。
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO StateRecord(
                student_id, session_id, record_time, avg_ear, avg_mar,
                avg_pitch, avg_yaw, avg_roll, perclos,
                attention_score, fatigue_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id, session_id, datetime.now().isoformat(),
                avg_ear, avg_mar, avg_pitch, avg_yaw, avg_roll,
                perclos, attention_score, fatigue_score,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_behavior(
        self,
        student_id: int,
        session_id: int,
        behavior_type: str,
        start_time: datetime,
        end_time: datetime,
        duration: float,
    ) -> int:
        """插入行为事件记录。

        Args:
            student_id: 学生 ID。
            session_id: 会话 ID。
            behavior_type: 行为类型。
            start_time: 开始时间。
            end_time: 结束时间。
            duration: 持续时间（秒）。

        Returns:
            记录 ID。
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO BehaviorRecord(
                student_id, session_id, behavior_type,
                start_time, end_time, duration
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                student_id, session_id, behavior_type,
                start_time.isoformat(), end_time.isoformat(), duration,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_warning(
        self,
        student_id: int,
        session_id: int,
        warning_level: str,
        warning_type: str,
        warning_reason: str,
        attention_score: float,
    ) -> int:
        """插入风险预警记录。

        Args:
            student_id: 学生 ID。
            session_id: 会话 ID。
            warning_level: 预警等级。
            warning_type: 预警类型。
            warning_reason: 预警原因。
            attention_score: 专注度评分。

        Returns:
            记录 ID。
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO WarningLog(
                student_id, session_id, warning_level, warning_type,
                warning_reason, attention_score, create_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id, session_id, warning_level, warning_type,
                warning_reason, attention_score, datetime.now().isoformat(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def query_state_records(
        self, student_id: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查询状态统计数据。

        Args:
            student_id: 学生 ID 过滤条件，None 表示所有学生。
            limit: 返回记录条数上限。

        Returns:
            状态记录字典列表。
        """
        query = "SELECT * FROM StateRecord"
        params: List[Any] = []
        if student_id is not None:
            query += " WHERE student_id = ?"
            params.append(student_id)
        query += " ORDER BY record_time DESC LIMIT ?"
        params.append(limit)
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def query_frame_records(
        self, student_id: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """\u67e5\u8be2\u539f\u59cb\u5e27\u89c6\u89c9\u6570\u636e\u3002

        Args:
            student_id: \u5b66\u751f ID \u8fc7\u6ee4\u6761\u4ef6\uff0cNone \u8868\u793a\u6240\u6709\u5b66\u751f\u3002
            limit: \u8fd4\u56de\u8bb0\u5f55\u6761\u6570\u4e0a\u9650\u3002

        Returns:
            \u5e27\u8bb0\u5f55\u5b57\u5178\u5217\u8868\u3002
        """
        query = "SELECT * FROM FrameRecord"
        params: List[Any] = []
        if student_id is not None:
            query += " WHERE student_id = ?"
            params.append(student_id)
        query += " ORDER BY frame_time DESC LIMIT ?"
        params.append(limit)
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def query_behavior_records(
        self, student_id: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """\u67e5\u8be2\u884c\u4e3a\u4e8b\u4ef6\u8bb0\u5f55\u3002

        Args:
            student_id: \u5b66\u751f ID \u8fc7\u6ee4\u6761\u4ef6\uff0cNone \u8868\u793a\u6240\u6709\u5b66\u751f\u3002
            limit: \u8fd4\u56de\u8bb0\u5f55\u6761\u6570\u4e0a\u9650\u3002

        Returns:
            \u884c\u4e3a\u8bb0\u5f55\u5b57\u5178\u5217\u8868\u3002
        """
        query = "SELECT * FROM BehaviorRecord"
        params: List[Any] = []
        if student_id is not None:
            query += " WHERE student_id = ?"
            params.append(student_id)
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def query_warning_records(
        self, student_id: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """\u67e5\u8be2\u98ce\u9669\u9884\u8b66\u8bb0\u5f55\u3002

        Args:
            student_id: \u5b66\u751f ID \u8fc7\u6ee4\u6761\u4ef6\uff0cNone \u8868\u793a\u6240\u6709\u5b66\u751f\u3002
            limit: \u8fd4\u56de\u8bb0\u5f55\u6761\u6570\u4e0a\u9650\u3002

        Returns:
            \u9884\u8b66\u8bb0\u5f55\u5b57\u5178\u5217\u8868\u3002
        """
        query = "SELECT * FROM WarningLog"
        params: List[Any] = []
        if student_id is not None:
            query += " WHERE student_id = ?"
            params.append(student_id)
        query += " ORDER BY create_time DESC LIMIT ?"
        params.append(limit)
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def set_config(self, key: str, value: str) -> None:
        """写入系统配置键值。

        Args:
            key: 配置键。
            value: 配置值。
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO SystemConfig(config_key, config_value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get_config(self, key: str) -> Optional[str]:
        """读取系统配置值。

        Args:
            key: 配置键。

        Returns:
            配置值；不存在时返回 None。
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT config_value FROM SystemConfig WHERE config_key = ?",
            (key,),
        )
        row = cursor.fetchone()
        return row["config_value"] if row else None
