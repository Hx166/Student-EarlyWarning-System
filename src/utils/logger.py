"""src/utils/logger.py —— 日志系统初始化模块。

根据 configs/logging.yaml 配置统一的日志输出，
支持控制台与文件（RotatingFileHandler）双输出。
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from src.utils.config import get_logging, PROJECT_ROOT

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 已初始化标记，避免重复添加 handler
_initialized = False


def _build_formatter(cfg: dict) -> logging.Formatter:
    """根据配置构造日志格式器。

    Args:
        cfg: 日志配置字典。

    Returns:
        配置好的 Formatter 实例。
    """
    fmt = cfg.get("format", _DEFAULT_FORMAT)
    date_fmt = cfg.get("date_format", _DEFAULT_DATE_FORMAT)
    return logging.Formatter(fmt=fmt, datefmt=date_fmt)


def _setup_file_handler(cfg: dict, formatter: logging.Formatter) -> Optional[RotatingFileHandler]:
    """创建文件日志处理器。

    Args:
        cfg: 日志配置字典。
        formatter: 日志格式器。

    Returns:
        配置好的 RotatingFileHandler，失败时返回 None。
    """
    if not cfg.get("file", True):
        return None

    log_dir = cfg.get("dir", "logs")
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(PROJECT_ROOT, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "system.log")
    handler = RotatingFileHandler(
        log_path,
        maxBytes=int(cfg.get("max_bytes", 10485760)),
        backupCount=int(cfg.get("backup_count", 5)),
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    return handler


def init_logging(level: Optional[str] = None) -> None:
    """初始化全局日志系统。

    Args:
        level: 覆盖日志级别；为 None 时使用配置文件中的级别。
    """
    global _initialized

    cfg = get_logging()
    root = logging.getLogger()
    root.setLevel(level or cfg.get("level", "INFO").upper())

    # 避免重复初始化导致 handler 堆积
    if _initialized:
        return

    formatter = _build_formatter(cfg)

    if cfg.get("console", True):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    file_handler = _setup_file_handler(cfg, formatter)
    if file_handler is not None:
        root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器。

    Args:
        name: 日志记录器名称（通常传 __name__）。

    Returns:
        配置好的 Logger 实例。
    """
    return logging.getLogger(name)
