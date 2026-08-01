"""src.camera —— 摄像头采集模块。

负责摄像头的连接、视频帧读取、FPS 统计与异常重连。
对外仅暴露 Camera 类与相关常量。
"""

from src.camera.camera import Camera, CameraError

__all__ = ["Camera", "CameraError"]
