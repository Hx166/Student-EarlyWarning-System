"""src/database/schema.py —— SQLite 数据库表结构定义。

定义第一阶段数据库的 DDL 语句与建表逻辑。
表结构严格遵循 docs/03-数据库/数据库设计.md 规范。
"""

from __future__ import annotations

import sqlite3
from typing import List

# 数据库建表语句
SCHEMA_STATEMENTS: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS Student(
        student_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_no TEXT UNIQUE,
        student_name TEXT,
        gender TEXT,
        class_name TEXT,
        major TEXT,
        create_time DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS CourseSession(
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_name TEXT,
        teacher_name TEXT,
        classroom TEXT,
        start_time DATETIME,
        end_time DATETIME
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS FrameRecord(
        frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        session_id INTEGER,
        frame_time DATETIME,
        ear REAL,
        mar REAL,
        pitch REAL,
        yaw REAL,
        roll REAL,
        FOREIGN KEY(student_id) REFERENCES Student(student_id),
        FOREIGN KEY(session_id) REFERENCES CourseSession(session_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS StateRecord(
        state_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        session_id INTEGER,
        record_time DATETIME,
        avg_ear REAL,
        avg_mar REAL,
        avg_pitch REAL,
        avg_yaw REAL,
        avg_roll REAL,
        perclos REAL,
        attention_score REAL,
        fatigue_score REAL,
        FOREIGN KEY(student_id) REFERENCES Student(student_id),
        FOREIGN KEY(session_id) REFERENCES CourseSession(session_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS BehaviorRecord(
        behavior_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        session_id INTEGER,
        behavior_type TEXT,
        start_time DATETIME,
        end_time DATETIME,
        duration REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS WarningLog(
        warning_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        session_id INTEGER,
        warning_level TEXT,
        warning_type TEXT,
        warning_reason TEXT,
        attention_score REAL,
        create_time DATETIME,
        processed INTEGER DEFAULT 0
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS SystemConfig(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_key TEXT,
        config_value TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS PredictionLog(
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        session_id INTEGER,
        record_time DATETIME,
        risk_label TEXT,
        risk_probability REAL,
        risk_level TEXT,
        model_version TEXT,
        feature_json TEXT,
        FOREIGN KEY(student_id) REFERENCES Student(student_id),
        FOREIGN KEY(session_id) REFERENCES CourseSession(session_id)
    );
    """,
]


def create_tables(conn: sqlite3.Connection) -> None:
    """创建数据库全部数据表。

    Args:
        conn: SQLite 数据库连接对象。

    Raises:
        sqlite3.Error: 建表失败时抛出。
    """
    cursor = conn.cursor()
    for statement in SCHEMA_STATEMENTS:
        cursor.execute(statement)
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    """创建数据库常用索引。

    Args:
        conn: SQLite 数据库连接对象。
    """
    cursor = conn.cursor()
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_student_no ON Student(student_no);",
        "CREATE INDEX IF NOT EXISTS idx_state_student ON StateRecord(student_id);",
        "CREATE INDEX IF NOT EXISTS idx_state_time ON StateRecord(record_time);",
        "CREATE INDEX IF NOT EXISTS idx_warning_student ON WarningLog(student_id);",
        "CREATE INDEX IF NOT EXISTS idx_frame_student ON FrameRecord(student_id);",
        "CREATE INDEX IF NOT EXISTS idx_frame_time ON FrameRecord(frame_time);",
        "CREATE INDEX IF NOT EXISTS idx_prediction_student ON PredictionLog(student_id);",
        "CREATE INDEX IF NOT EXISTS idx_prediction_time ON PredictionLog(record_time);",
    ]
    for statement in indexes:
        cursor.execute(statement)
    conn.commit()
