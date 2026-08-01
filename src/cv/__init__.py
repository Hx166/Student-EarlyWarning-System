"""src.cv —— 计算机视觉分析包。

负责人脸检测、Face Mesh、眼部状态、头部姿态、
疲劳评估与课堂参与度分析等视觉处理流程。

对外统一暴露 CVProcessor 作为视觉处理入口。
"""

from src.cv.processor import CVProcessor

__all__ = ["CVProcessor"]
