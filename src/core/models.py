"""src/core/models.py —— 系统统一数据结构定义。

本模块集中定义系统中各模块之间传递的数据对象。
为保证模块低耦合，任何模块之间不允许直接操作对方的内部对象，
而应统一通过本模块定义的数据结构进行通信。

数据结构遵循 docs/04-CV模块 各设计文档中定义的标准字段，
确保接口稳定、可扩展。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Frame:
    """单帧视频数据。

    Attributes:
        frame_id: 帧编号（自增流水号）。
        timestamp: 帧采集时间。
        image: OpenCV BGR 图像矩阵。
        width: 图像宽度。
        height: 图像高度。
        extra: 附加扩展信息。
    """

    frame_id: int
    timestamp: datetime
    image: np.ndarray
    width: int = 0
    height: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """图像元信息初始化（若未显式给出则从矩阵推断）。"""
        if self.image is not None and self.image.size > 0:
            height, width = self.image.shape[:2]
            self.height = height if self.height == 0 else self.height
            self.width = width if self.width == 0 else self.width

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。

        Returns:
            包含帧基本信息的字典（不含图像矩阵）。
        """
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "width": self.width,
            "height": self.height,
        }


@dataclass
class FaceLandmark:
    """单张人脸的 MediaPipe Face Mesh 关键点数据。

    Attributes:
        face_id: 人脸编号。
        confidence: 检测置信度。
        bbox: 人脸边界框 (x, y, w, h)。
        center_x: 人脸中心 X 像素坐标。
        center_y: 人脸中心 Y 像素坐标。
        landmarks: 468 个关键点的归一化坐标列表 [(x, y, z), ...]。
        visibility: 关键点可见度。
        roi: 人脸区域图像。
    """

    face_id: int
    confidence: float
    bbox: Optional[tuple] = None
    center_x: float = 0.0
    center_y: float = 0.0
    landmarks: Optional[List[Any]] = None
    visibility: Optional[List[float]] = None
    roi: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。

        Returns:
            人脸关键点信息字典。
        """
        return {
            "face_id": self.face_id,
            "confidence": self.confidence,
            "bbox": list(self.bbox) if self.bbox else None,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "landmarks": self.landmarks,
        }


@dataclass
class PoseLandmark:
    """人体姿态关键点数据。

    Attributes:
        landmarks: 33 个关键点坐标列表。
        visibility: 各关键点可见度。
        confidence: 姿态检测置信度。
    """

    landmarks: Optional[List[Any]] = None
    visibility: Optional[List[float]] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。

        Returns:
            姿态关键点信息字典。
        """
        return {
            "landmarks": self.landmarks,
            "visibility": self.visibility,
            "confidence": self.confidence,
        }


@dataclass
class EyeState:
    """眼部状态检测结果（EyeEAR 模块输出）。

    Attributes:
        left_ear: 左眼纵横比。
        right_ear: 右眼纵横比。
        average_ear: 平均纵横比。
        eye_state: 睁眼/闭眼状态字符串。
        blink: 是否发生眨眼。
        perclos: 闭眼时间占比 (0~1)。
        confidence: 可信度。
    """

    left_ear: float = 0.0
    right_ear: float = 0.0
    average_ear: float = 0.0
    eye_state: str = "unknown"
    blink: bool = False
    perclos: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """转换为可序列化字典。

        Returns:
            眼部状态结果字典。
        """
        return {
            "left_ear": self.left_ear,
            "right_ear": self.right_ear,
            "average_ear": self.average_ear,
            "eye_state": self.eye_state,
            "blink": self.blink,
            "perclos": self.perclos,
            "confidence": self.confidence,
        }


@dataclass
class MouthState:
    """嘴部状态检测结果（Mouth MAR 模块输出）。

    Attributes:
        mar: 嘴部纵横比。
        is_yawn: 是否判定为打哈欠。
        confidence: 可信度。
    """

    mar: float = 0.0
    is_yawn: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。

        Returns:
            嘴部状态结果字典。
        """
        return {
            "mar": self.mar,
            "is_yawn": self.is_yawn,
            "confidence": self.confidence,
        }


@dataclass
class HeadPoseResult:
    """头部姿态估计结果（HeadPose 模块输出）。

    Attributes:
        pitch: 俯仰角（度）。
        yaw: 偏航角（度）。
        roll: 翻滚角（度）。
        pose_status: 姿态分类（Normal/LookDown/LookLeft/LookRight/Tilt）。
        confidence: 可信度。
    """

    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    pose_status: str = "Normal"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。

        Returns:
            头部姿态结果字典。
        """
        return {
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "pose_status": self.pose_status,
            "confidence": self.confidence,
        }


@dataclass
class FatigueResult:
    """疲劳状态评估结果（Fatigue 模块输出）。

    Attributes:
        perclos: 闭眼时间占比。
        blink_rate: 眨眼频率（次/分钟）。
        head_down_duration: 累计低头时间（秒）。
        yawn_rate: 打哈欠频率（次/分钟）。
        yawn_duration: 累计打哈欠时间（秒）。
        fatigue_score: 综合疲劳评分 (0~100)。
        fatigue_level: 疲劳等级字符串。
        confidence: 可信度。
    """

    perclos: float = 0.0
    blink_rate: float = 0.0
    head_down_duration: float = 0.0
    yawn_rate: float = 0.0
    yawn_duration: float = 0.0
    fatigue_score: float = 0.0
    fatigue_level: str = "normal"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """转换为可序列化字典。

        Returns:
            疲劳评估结果字典。
        """
        return {
            "perclos": self.perclos,
            "blink_rate": self.blink_rate,
            "head_down_duration": self.head_down_duration,
            "yawn_rate": self.yawn_rate,
            "yawn_duration": self.yawn_duration,
            "fatigue_score": self.fatigue_score,
            "fatigue_level": self.fatigue_level,
            "confidence": self.confidence,
        }


@dataclass
class EngagementResult:
    """课堂参与度评估结果（Engagement 模块输出）。

    Attributes:
        engagement_score: 课堂参与度评分 (0~100)。
        engagement_level: 参与度等级字符串。
        attention_state: 当前关注状态字符串。
        update_time: 更新时间。
        confidence: 可信度。
        attention_score: 专注度评分别名（与 engagement_score 一致）。
    """

    engagement_score: float = 0.0
    engagement_level: str = "normal"
    attention_state: str = "unknown"
    update_time: Optional[datetime] = None
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """默认时间与状态初始化。"""
        if self.update_time is None:
            self.update_time = datetime.now()
        if self.attention_state == "unknown":
            self.attention_state = self.engagement_level

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。

        Returns:
            参与度评估结果字典。
        """
        return {
            "engagement_score": self.engagement_score,
            "engagement_level": self.engagement_level,
            "attention_state": self.attention_state,
            "update_time": (
                self.update_time.isoformat() if self.update_time else None
            ),
            "confidence": self.confidence,
        }


@dataclass
class BehaviorResult:
    """课堂行为识别结果。

    Attributes:
        behavior_type: 行为类型（blink/yawn/head_down/looking_away/absent）。
        start_time: 行为开始时间。
        end_time: 行为结束时间。
        duration: 行为持续时间（秒）。
        confidence: 可信度。
        student_id: 关联学生 ID（可选）。
    """

    behavior_type: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    confidence: float = 0.0
    student_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。

        Returns:
            行为结果字典。
        """
        return {
            "behavior_type": self.behavior_type,
            "start_time": (
                self.start_time.isoformat() if self.start_time else None
            ),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "confidence": self.confidence,
            "student_id": self.student_id,
        }

@dataclass
class LSTMResult:
    """LSTM 时序预测结果（LSTM 模块输出）。

    Attributes:
        prediction: 预测类别（normal/attention_down/fatigue/high_risk）。
        probability: 预测类别概率（0~1）。
        trend: 状态变化趋势（stable/declining/deteriorating）。
        confidence: 预测置信度（与 probability 一致）。
        prediction_time: 预测时间。
        extra: 扩展信息。
    """

    prediction: str = "normal"
    probability: float = 0.0
    trend: str = "stable"
    confidence: float = 0.0
    prediction_time: Optional[datetime] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """默认时间与置信度初始化。"""
        if self.prediction_time is None:
            self.prediction_time = datetime.now()
        if self.confidence == 0.0:
            self.confidence = self.probability

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。

        Returns:
            LSTM 预测结果字典。
        """
        return {
            "prediction": self.prediction,
            "probability": self.probability,
            "trend": self.trend,
            "confidence": self.confidence,
            "prediction_time": (
                self.prediction_time.isoformat() if self.prediction_time else None
            ),
            "extra": self.extra,
        }


@dataclass
class RiskPredictionResult:
    """学业风险预测结果（Risk Prediction 模块输出）。

    Attributes:
        risk_label: 4 类状态标签（normal/attention_down/fatigue/high_risk）。
        risk_probability: 预测类别概率（0~1，即置信度）。
        risk_level: 风险等级（normal/attention/warning/high_risk）。
        confidence: 预测置信度（与 risk_probability 一致）。
        model_version: 使用的模型版本。
        prediction_time: 预测时间。
        extra: 扩展信息。
    """

    risk_label: str = "normal"
    risk_probability: float = 0.0
    risk_level: str = "normal"
    confidence: float = 0.0
    model_version: str = "v1"
    prediction_time: Optional[datetime] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """默认时间与置信度初始化。"""
        if self.prediction_time is None:
            self.prediction_time = datetime.now()
        if self.confidence == 0.0:
            self.confidence = self.risk_probability

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。

        Returns:
            风险预测结果字典。
        """
        return {
            "risk_label": self.risk_label,
            "risk_probability": self.risk_probability,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "prediction_time": (
                self.prediction_time.isoformat() if self.prediction_time else None
            ),
            "extra": self.extra,
        }


@dataclass
class FaceAnalysis:
    """单张人脸的完整视觉分析结果（多人逐人分析）。

    Attributes:
        face_id: 人脸编号。
        eye: 眼部状态结果。
        head_pose: 头部姿态结果。
        mouth: 嘴部状态结果。
        fatigue: 疲劳评估结果。
        engagement: 参与度评估结果。
    """

    face_id: int = 0
    eye: Optional[EyeState] = None
    head_pose: Optional[HeadPoseResult] = None
    mouth: Optional[MouthState] = None
    fatigue: Optional[FatigueResult] = None
    engagement: Optional[EngagementResult] = None

    def __post_init__(self) -> None:
        """默认空结果初始化。"""
        if self.eye is None:
            self.eye = EyeState()
        if self.head_pose is None:
            self.head_pose = HeadPoseResult()
        if self.mouth is None:
            self.mouth = MouthState()
        if self.fatigue is None:
            self.fatigue = FatigueResult()
        if self.engagement is None:
            self.engagement = EngagementResult()
