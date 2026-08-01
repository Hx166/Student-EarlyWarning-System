"""src.api —— Web API 服务模块。

基于 Flask 提供 REST 接口，供前端 Dashboard 消费。
"""

from src.api.app import create_app

__all__ = ["create_app"]
