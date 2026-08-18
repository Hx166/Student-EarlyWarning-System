"""src/database/timeseries.py —— 高频时序数据写入器（InfluxDB 预留）。

为 FrameRecord / StateRecord 等高频数据预留 InfluxDB 接入能力，
通过 configs/database.yaml 的 influxdb.enabled 开关控制（默认关闭）。
启用时才惰性导入 influxdb_client。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InfluxDBWriter:
    """InfluxDB 高频时序数据写入器（预留，默认关闭）。

    Attributes:
        url: InfluxDB 地址。
        token: 访问令牌。
        org: 组织名。
        bucket: 桶名。
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """初始化 InfluxDB 写入器。

        Args:
            cfg: influxdb 配置字典。
        """
        self.url = cfg.get("url", "http://localhost:8086")
        self.token = cfg.get("token", "")
        self.org = cfg.get("org", "")
        self.bucket = cfg.get("bucket", "early_warning")
        self._client: Optional[Any] = None

    def _ensure_client(self) -> Any:
        """惰性创建 InfluxDB 客户端。

        Returns:
            InfluxDBClient 实例。

        Raises:
            ImportError: influxdb_client 未安装时抛出。
        """
        if self._client is None:
            try:
                from influxdb_client import InfluxDBClient
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "未安装 influxdb-client，无法使用 InfluxDB，请执行 pip install influxdb-client"
                ) from exc
            self._client = InfluxDBClient(
                url=self.url, token=self.token, org=self.org
            )
        return self._client

    def write_frame_record(self, fields: Dict[str, Any]) -> None:
        """写入单帧视觉数据（预留实现）。

        Args:
            fields: 帧记录字段字典。
        """
        self._write("frame_record", fields)

    def write_state_record(self, fields: Dict[str, Any]) -> None:
        """写入每秒状态数据（预留实现）。

        Args:
            fields: 状态记录字段字典。
        """
        self._write("state_record", fields)

    def _write(self, measurement: str, fields: Dict[str, Any]) -> None:
        """写入测量点。

        Args:
            measurement: 测量名。
            fields: 字段字典。
        """
        from influxdb_client import Point

        client = self._ensure_client()
        write_api = client.write_api()
        point = Point(measurement)
        for key, value in fields.items():
            point.field(key, value)
        write_api.write(bucket=self.bucket, org=self.org, record=point)
        logger.debug("InfluxDB 写入 %s", measurement)


def create_timeseries_writer(cfg: Dict[str, Any]) -> Optional[InfluxDBWriter]:
    """根据配置创建时序写入器。

    Args:
        cfg: influxdb 配置字典。

    Returns:
        写入器实例；未启用时返回 None。
    """
    if not cfg.get("enabled", False):
        return None
    return InfluxDBWriter(cfg)
