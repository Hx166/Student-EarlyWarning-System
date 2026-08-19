"""src/main.py —— 系统启动入口。

负责初始化日志、配置、数据库、视觉流水线、Flask 服务，
并启动系统主流程。
"""

from __future__ import annotations

import argparse
import threading

from src.config import get_system
from src.utils.logger import get_logger, init_logging

# 先初始化日志再加载模块日志器
init_logging()
logger = get_logger(__name__)


def _load_config():
    """加载系统配置。

    Returns:
        系统配置字典。
    """
    return get_system()


def _setup_components():
    """初始化各核心组件。

    Returns:
        (db, camera, cv, pipeline) 组件元组。

    Raises:
        Exception: 组件初始化失败时抛出。
    """
    from src.camera import Camera
    from src.cv import CVProcessor
    from src.database import DatabaseService
    from src.service import PipelineService

    defaults = get_system().get("defaults", {})

    db = DatabaseService().connect()

    # 插入默认学生与会话
    student_id = db.upsert_student(
        student_no=defaults.get("student_no", "S00001"),
        student_name=defaults.get("student_name", "测试学生"),
        class_name=defaults.get("class_name", ""),
        major=defaults.get("major", ""),
    )
    session_id = db.create_session(
        course_name=defaults.get("course_name", "课堂状态测试"),
        teacher_name=defaults.get("teacher_name", "测试教师"),
        classroom=defaults.get("classroom", ""),
    )

    camera = Camera().open()
    cv = CVProcessor()
    pipeline = PipelineService(
        camera=camera, cv=cv, db=db,
        student_id=student_id, session_id=session_id,
    )
    return db, camera, cv, pipeline


def _run_flask(host: str, port: int, debug: bool, database=None, pipeline=None):
    """启动 Flask API 服务。

    Args:
        host: 监听地址。
        port: 监听端口。
        debug: 是否开启调试模式。
        database: 可选数据库服务，注入 Flask 供查询接口使用。
        pipeline: 可选主流程服务，注入 Flask 供运行指标接口使用。
    """
    from src.api import create_app

    app = create_app(database=database, pipeline=pipeline)
    logger.info("Flask 服务启动于 %s:%s", host, port)
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    """系统主函数。"""
    parser = argparse.ArgumentParser(description="学生学业状态分析与预警系统")
    parser.add_argument("--frames", type=int, default=None,
                        help="最大处理帧数，默认无限运行")
    parser.add_argument("--api-only", action="store_true",
                        help="仅启动 API 服务，不启动摄像头采集")
    args = parser.parse_args()

    cfg = _load_config()
    server = cfg.get("server", {})

    if args.api_only:
        logger.info("以 API-only 模式启动")
        from src.database import DatabaseService

        db = DatabaseService().connect()
        try:
            _run_flask(server.get("host", "0.0.0.0"), server.get("port", 5000),
                       server.get("debug", False), database=db)
        finally:
            db.close()
        return

    db, camera, cv, pipeline = _setup_components()

    # 后台线程启动 Flask
    flask_thread = threading.Thread(
        target=_run_flask,
        args=(server.get("host", "0.0.0.0"), server.get("port", 5000),
              server.get("debug", False), db, pipeline),
        daemon=True,
    )
    flask_thread.start()
    logger.info("Flask 服务后台线程已启动")

    try:
        pipeline.run(max_frames=args.frames)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出...")
    finally:
        pipeline.close()
        db.close()


if __name__ == "__main__":
    main()
