"""scripts/migrate_sqlite_to_mysql.py —— SQLite → MySQL 数据迁移脚本。

将 data/early_warning.db（SQLite）中的数据迁移到 MySQL（配置见 configs/database.yaml）。

用法：
    python scripts/migrate_sqlite_to_mysql.py [--source data/early_warning.db]

依赖：pymysql（pip install pymysql）。
迁移前请确保 MySQL 中已创建目标数据库（默认 early_warning）。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

# 将项目根目录加入 sys.path，以便导入 src 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import get_database
from src.database.backends import MySQLBackend, _resolve_sqlite_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 迁移顺序（先无外键依赖，再外键依赖表）
TABLES = [
    "Student",
    "CourseSession",
    "SystemConfig",
    "FrameRecord",
    "StateRecord",
    "BehaviorRecord",
    "WarningLog",
    "PredictionLog",
]


def _normalize(value):
    """将 ISO 时间字符串转为 MySQL DATETIME 可接受格式。

    Args:
        value: 原始值。

    Returns:
        规整后的值。
    """
    if isinstance(value, str) and len(value) > 10 and value[10] == "T":
        return value.replace("T", " ")
    return value


def read_sqlite(path: str):
    """读取 SQLite 全部表数据。

    Args:
        path: SQLite 数据库文件路径。

    Returns:
        表名到 (列名列表, 行字典列表) 的映射。
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    data = {}
    for table in TABLES:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table}")
        columns = [d[0] for d in cursor.description]
        rows = [[_normalize(row[c]) for c in columns] for row in cursor.fetchall()]
        data[table] = (columns, rows)
    conn.close()
    return data


def write_mysql(cfg, data) -> int:
    """将数据写入 MySQL。

    Args:
        cfg: MySQL 配置字典。
        data: 表名到 (列名, 行) 的映射。

    Returns:
        写入的总行数。
    """
    backend = MySQLBackend(cfg)
    conn = backend.connect()
    backend.create_schema(conn)
    total = 0
    for table in TABLES:
        columns, rows = data[table]
        if not rows:
            continue
        placeholders = ", ".join(["%s"] * len(columns))
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"VALUES ({placeholders})"
        )
        cursor = conn.cursor()
        for row in rows:
            cursor.execute(sql, row)
        conn.commit()
        total += len(rows)
        logger.info("已迁移表 %s，共 %s 行", table, len(rows))
    conn.close()
    return total


def main() -> int:
    """迁移主函数。

    Returns:
        0 表示成功。
    """
    parser = argparse.ArgumentParser(description="SQLite → MySQL 数据迁移")
    parser.add_argument(
        "--source", default=None,
        help="SQLite 数据库文件路径，默认读取配置中的 sqlite url",
    )
    args = parser.parse_args()

    cfg = get_database()
    source = args.source or _resolve_sqlite_path(
        cfg.get("database", {}).get("url", "sqlite:///./data/early_warning.db")
    )
    if not os.path.exists(source):
        logger.error("SQLite 数据库不存在: %s", source)
        return 1

    data = read_sqlite(source)
    total = write_mysql(cfg.get("mysql", {}), data)
    logger.info("迁移完成，共写入 %s 行", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
