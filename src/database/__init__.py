"""src.database —— 数据库存储模块。

基于 SQLite 实现系统数据持久化，
对外暴露 DatabaseService 作为统一访问入口。
"""

from src.database.service import DatabaseService

__all__ = ["DatabaseService"]
