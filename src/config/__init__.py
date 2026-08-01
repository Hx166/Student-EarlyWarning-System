"""src.config —— 配置门面（Facade）包。

对外统一暴露配置加载接口，内部委托给 src.utils.config 实现。
业务模块可通过本门面获取配置，避免直接依赖底层实现细节。
"""

from src.utils.config import (
    ConfigError,
    clear_cache,
    get_camera,
    get_database,
    get_logging,
    get_system,
    load_config,
)

__all__ = [
    "ConfigError",
    "clear_cache",
    "get_camera",
    "get_database",
    "get_logging",
    "get_system",
    "load_config",
]
