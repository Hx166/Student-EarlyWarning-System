"""src.cv.mediapipe —— MediaPipe 视觉关键点检测模块。

基于 MediaPipe 提供人脸 Face Mesh（468 关键点）与人体 Pose（33 关键点）
检测能力，将原始模型输出统一转换为项目内部数据结构。
"""

from src.cv.mediapipe.processor import MediaPipeProcessor, MediaPipeResult

__all__ = ["MediaPipeProcessor", "MediaPipeResult"]
