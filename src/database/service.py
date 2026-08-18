"""src/database/service.py —— 数据库统一访问服务。

提供数据库连接的建立、表的初始化以及常用数据的写入与查询能力。
默认使用 SQLite 后端，可通过 configs/database.yaml 的 backend 切换为 MySQL。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import get_database
from src.database.backends import DatabaseBackend, create_backend
from src.database.prediction import PredictionMixin
from src.database.timeseries import create_timeseries_writer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseService(PredictionMixin):
    """数据库访问服务。

    负责连接管理与数据持久化操作，后端可切换（SQLite 默认 / MySQL）。

    Attributes:
        conn: 底层连接对象。
        _backend: 存储后端。
        ts_writer: 高频时序写入器（InfluxDB 预留，可为 None）。
    """

    def __init__(
        self, db_path: Optional[str] = None, backend: Optional[str] = None
    ) -> None:
        """初始化数据库服务。

        Args:
            db_path: SQLite 数据库文件路径；为 None 时从配置读取。
            backend: 后端名称（sqlite/mysql）；为 None 时从配置读取。
        """
        cfg = get_database()
        backend_name = backend or cfg.get("database", {}).get("backend", "sqlite")
        self._backend: DatabaseBackend = create_backend(backend_name, cfg, db_path)
        self.conn: Optional[Any] = None
        self.ts_writer = create_timeseries_writer(cfg.get("influxdb", {}))

    def connect(self) -> "DatabaseService":
        """建立数据库连接并初始化表结构。

        Returns:
            当前实例，便于链式调用。
        """
        self.conn = self._backend.connect()
        self._backend.create_schema(self.conn)
        logger.info("数据库连接成功，后端=%s", self._backend.__class__.__name__)
        return self

    def close(self) -> None:
        """关闭数据库连接。"""
        if self.conn is not None:
            self._backend.close(self.conn)
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
        row = self._backend.query_one(
            self.conn,
            "SELECT student_id FROM Student WHERE student_no = ?",
            (student_no,),
        )
        if row is not None:
            self._backend.execute(
                self.conn,
                """
                UPDATE Student SET student_name=?, gender=?, class_name=?, major=?
                WHERE student_no=?
                """,
                (student_name, gender, class_name, major, student_no),
            )
            return int(row["student_id"])
        return self._backend.execute(
            self.conn,
            """
            INSERT INTO Student(student_no, student_name, gender, class_name, major)
            VALUES (?, ?, ?, ?, ?)
            """,
            (student_no, student_name, gender, class_name, major),
        )

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
        return self._backend.execute(
            self.conn,
            """
            INSERT INTO CourseSession(course_name, teacher_name, classroom, start_time)
            VALUES (?, ?, ?, ?)
            """,
            (course_name, teacher_name, classroom, start_time.isoformat()),
        )

    def end_session(self, session_id: int) -> None:
        """结束课堂会话。

        Args:
            session_id: 会话 ID。
        """
        self._backend.execute(
            self.conn,
            "UPDATE CourseSession SET end_time=? WHERE session_id=?",
            (datetime.now().isoformat(), session_id),
        )

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
        record_id = self._backend.execute(
            self.conn,
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
        self._write_timeseries("frame_record", {
            "student_id": student_id, "session_id": session_id,
            "ear": ear, "mar": mar, "pitch": pitch, "yaw": yaw, "roll": roll,
        })
        return record_id

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
        record_id = self._backend.execute(
            self.conn,
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
        self._write_timeseries("state_record", {
            "student_id": student_id, "session_id": session_id,
            "avg_ear": avg_ear, "avg_mar": avg_mar, "perclos": perclos,
            "attention_score": attention_score, "fatigue_score": fatigue_score,
        })
        return record_id

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
        return self._backend.execute(
            self.conn,
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
        return self._backend.execute(
            self.conn,
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
        return self._backend.query(self.conn, query, params)

    def query_frame_records(
        self, student_id: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查询原始帧视觉数据。

        Args:
            student_id: 学生 ID 过滤条件，None 表示所有学生。
            limit: 返回记录条数上限。

        Returns:
            帧记录字典列表。
        """
        query = "SELECT * FROM FrameRecord"
        params: List[Any] = []
        if student_id is not None:
            query += " WHERE student_id = ?"
            params.append(student_id)
        query += " ORDER BY frame_time DESC LIMIT ?"
        params.append(limit)
        return self._backend.query(self.conn, query, params)

    def query_behavior_records(
        self, student_id: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查询行为事件记录。

        Args:
            student_id: 学生 ID 过滤条件，None 表示所有学生。
            limit: 返回记录条数上限。

        Returns:
            行为记录字典列表。
        """
        query = "SELECT * FROM BehaviorRecord"
        params: List[Any] = []
        if student_id is not None:
            query += " WHERE student_id = ?"
            params.append(student_id)
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        return self._backend.query(self.conn, query, params)

    def query_warning_records(
        self, student_id: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查询风险预警记录。

        Args:
            student_id: 学生 ID 过滤条件，None 表示所有学生。
            limit: 返回记录条数上限。

        Returns:
            预警记录字典列表。
        """
        query = "SELECT * FROM WarningLog"
        params: List[Any] = []
        if student_id is not None:
            query += " WHERE student_id = ?"
            params.append(student_id)
        query += " ORDER BY create_time DESC LIMIT ?"
        params.append(limit)
        return self._backend.query(self.conn, query, params)

    def set_config(self, key: str, value: str) -> None:
        """写入系统配置键值。

        Args:
            key: 配置键。
            value: 配置值。
        """
        self._backend.execute(
            self.conn,
            "INSERT INTO SystemConfig(config_key, config_value) VALUES (?, ?)",
            (key, value),
        )

    def get_config(self, key: str) -> Optional[str]:
        """读取系统配置值。

        Args:
            key: 配置键。

        Returns:
            配置值；不存在时返回 None。
        """
        row = self._backend.query_one(
            self.conn,
            "SELECT config_value FROM SystemConfig WHERE config_key = ?",
            (key,),
        )
        return row["config_value"] if row else None

    def _write_timeseries(self, kind: str, fields: Dict[str, Any]) -> None:
        """写入高频时序数据（InfluxDB 预留，关闭时无操作）。

        Args:
            kind: 写入类型（frame_record / state_record）。
            fields: 字段字典。
        """
        if self.ts_writer is None:
            return
        try:
            if kind == "frame_record":
                self.ts_writer.write_frame_record(fields)
            else:
                self.ts_writer.write_state_record(fields)
        except Exception as exc:  # 时序写入失败不影响主流程
            logger.warning("时序数据写入失败: %s", exc)
