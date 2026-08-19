# Bug 记录

## 说明

Bug 修复需：定位原因、修复代码、补充回归测试（对应 `tests/` 下用例）。

## 历史记录

| 编号 | 现象 | 原因 | 修复 | 回归测试 |
|------|------|------|------|----------|
| B001 | api-only 与主流程模式数据库未连接 | 数据库服务未注入 Flask | `create_app(database=...)` 注入 | tests/api |
| B002 | 多个视觉模块重复初始化 MediaPipe | 各自创建引擎 | 单实例依赖注入共享 | tests/cv |
| B003 | 摄像头重复打印日志 | 重复日志语句 | 移除冗余 logger | tests/camera |
| B004 | 数据库服务单文件超 500 行 | 预测方法并入 service.py | 抽离 PredictionMixin | tests/database |

## 待观察

- MySQL 后端在重连时索引重复创建（已 try/except 忽略，需长期验证）。
- InfluxDB 写入失败仅记日志，不影响主流程，需监控日志告警。
