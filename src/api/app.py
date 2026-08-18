"""src/api/app.py —— Flask 应用工厂。

创建并配置 Flask 应用，注册路由与蓝图，
作为系统对外提供的 REST API 服务。
"""

from __future__ import annotations

from typing import Callable, Optional

from flask import Flask

from src.config import get_system
from src.database import DatabaseService
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 运行时数据库服务引用（由 main 注入）
db_service: Optional[DatabaseService] = None


def create_app(database: Optional[DatabaseService] = None) -> Flask:
    """创建并配置 Flask 应用。

    Args:
        database: 数据库服务实例；None 时使用全局引用。

    Returns:
        配置完成的 Flask 应用实例。
    """
    global db_service
    if database is not None:
        db_service = database

    cfg = get_system()
    app = Flask(
        __name__,
        static_folder="../frontend/static",
        template_folder="../frontend/templates",
    )
    app.config["JSON_AS_ASCII"] = False

    _register_routes(app)
    logger.info("Flask 应用创建完成")
    return app


def _register_routes(app: Flask) -> None:
    """注册全部路由。

    Args:
        app: Flask 应用实例。
    """
    from src.api.routes import dashboard, metrics, predict, warning

    app.register_blueprint(dashboard.bp)
    app.register_blueprint(metrics.bp)
    app.register_blueprint(warning.bp)
    app.register_blueprint(predict.bp)
