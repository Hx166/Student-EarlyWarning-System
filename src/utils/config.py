"""src/utils/config.py —— 配置管理模块。

负责从 configs/ 目录加载 YAML 配置文件，
并向后端各模块提供统一的配置访问接口。

本模块属于通用工具，不依赖任何业务模块。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

# 项目根目录（src 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

# 已加载的配置缓存
_cache: Dict[str, Dict[str, Any]] = {}


class ConfigError(Exception):
    """配置加载异常。

    Raises:
        当配置文件不存在或格式错误时抛出。
    """


def load_config(name: str) -> Dict[str, Any]:
    """加载指定名称的配置文件。

    Args:
        name: 配置文件名称（不含扩展名），如 "system"。

    Returns:
        解析后的配置字典。

    Raises:
        ConfigError: 配置文件不存在或 YAML 解析失败。
    """
    if name in _cache:
        return _cache[name]

    path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise ConfigError(f"配置文件不存在: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件解析失败: {path}: {exc}") from exc

    _cache[name] = data
    return data


def get_system() -> Dict[str, Any]:
    """获取系统全局配置。

    Returns:
        系统配置字典。
    """
    return load_config("system")


def get_camera() -> Dict[str, Any]:
    """获取摄像头配置。

    Returns:
        摄像头配置字典。
    """
    return load_config("camera")


def get_database() -> Dict[str, Any]:
    """获取数据库配置。

    Returns:
        数据库配置字典。
    """
    return load_config("database")


def get_logging() -> Dict[str, Any]:
    """获取日志配置。

    Returns:
        日志配置字典。
    """
    return load_config("logging")


def clear_cache() -> None:
    """清空配置缓存（主要用于测试）。"""
    _cache.clear()
