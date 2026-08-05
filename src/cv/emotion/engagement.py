"""src/cv/emotion/engagement.py —— 课堂参与度（Engagement）评估模块。

融合眼部（EAR/PERCLOS/Blink）、头部姿态（Pitch/Yaw/Roll）、
疲劳与面部在场情况计算学生课堂参与度评分。

评分依据 docs/04-CV模块/Engagement.md 设计规范，
采用加权融合方法：
    engagement_score = 0.35 × EyeScore + 0.25 × HeadPoseScore
                      + 0.25 × FatigueScore + 0.15 × AttendanceScore
"""

from __future__ import annotations

from typing import Optional

from src.core.models import (
    EngagementResult,
    EyeState,
    FatigueResult,
    HeadPoseResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EngagementAnalyzer:
    """课堂参与度评估器。

    融合多质特征计算参与度评分，
    并输出参与度等级与当前注意状态。

    Attributes:
        attention_weights: 各特征组件权重。
    """

    def __init__(self) -> None:
        """初始化参与度评估器。"""
        self.attention_weights = {
            "eye": 0.35,
            "head_pose": 0.25,
            "fatigue": 0.25,
            "attendance": 0.15,
        }
        # 缺失输入时的中性基准分
        self._neutral_score = 60.0
        logger.info("EngagementAnalyzer 已初始化")

    def analyze(
        self,
        eye_state: Optional[EyeState] = None,
        head_pose: Optional[HeadPoseResult] = None,
        fatigue: Optional[FatigueResult] = None,
    ) -> EngagementResult:
        """综合评估课堂参与度。

        Args:
            eye_state: 眼部状态结果（可为 None）。
            head_pose: 头部姿态结果（可为 None）。
            fatigue: 疲劳评估结果（可为 None）。

        Returns:
            参与度评估结果对象。
        """
        eye_score = self._eye_score(eye_state)
        head_pose_score = self._head_pose_score(head_pose)
        fatigue_score_comp = self._fatigue_component(fatigue)
        attendance_score = self._attendance_score(eye_state, head_pose)

        w = self.attention_weights
        engagement_score = float(round(
            w["eye"] * eye_score
            + w["head_pose"] * head_pose_score
            + w["fatigue"] * fatigue_score_comp
            + w["attendance"] * attendance_score,
            2,
        ))

        engagement_level = self._classify_level(engagement_score)
        attention_state = self._classify_state(engagement_score)
        confidence = self._compute_confidence(eye_state, head_pose, fatigue)

        return EngagementResult(
            engagement_score=engagement_score,
            engagement_level=engagement_level,
            attention_state=attention_state,
            confidence=confidence,
        )

    @staticmethod
    def _eye_score(eye_state: Optional[EyeState]) -> float:
        """计算眼部组件评分。

        Args:
            eye_state: 眼部状态结果（可为 None）。

        Returns:
            眼部评分（0~100）。
        """
        if eye_state is None:
            return 60.0
        # 闭眼比例越高，眼部评分越低
        perclos = min(max(getattr(eye_state, "perclos", 0.0), 0.0), 1.0)
        return float(round(100.0 * (1.0 - perclos), 2))

    @staticmethod
    def _head_pose_score(head_pose: Optional[HeadPoseResult]) -> float:
        """计算头部姿态组件评分。

        Args:
            head_pose: 头部姿态结果（可为 None）。

        Returns:
            头部姿态评分（0~100）。
        """
        if head_pose is None or head_pose.confidence <= 0.0:
            return 60.0
        status = head_pose.pose_status
        # 低头与偏头显著降低参与度
        if status == "LookDown":
            return 45.0
        if status in ("LookLeft", "LookRight", "Tilt"):
            return 70.0
        return 100.0

    @staticmethod
    def _fatigue_component(fatigue: Optional[FatigueResult]) -> float:
        """计算疲劳对参与度的贡献分。

        Args:
            fatigue: 疲劳评估结果（可为 None）。

        Returns:
            疲劳组件评分（越高越好，0~100）。
        """
        if fatigue is None:
            return 60.0
        # 疲劳评分越高，参与度贡献越低
        fscore = min(max(getattr(fatigue, "fatigue_score", 0.0), 0.0), 100.0)
        return float(round(100.0 - fscore, 2))

    @staticmethod
    def _attendance_score(
        eye_state: Optional[EyeState],
        head_pose: Optional[HeadPoseResult],
    ) -> float:
        """计算面部在场组件评分。

        Args:
            eye_state: 眼部状态结果（可为 None）。
            head_pose: 头部姿态结果（可为 None）。

        Returns:
            在场评分（0~100）。
        """
        present = False
        if head_pose is not None and head_pose.confidence > 0.0:
            present = True
        if eye_state is not None and eye_state.confidence > 0.0:
            present = True
        # 长时间无人脸视为缺勤性降分
        return 100.0 if present else 20.0

    @staticmethod
    def _classify_level(score: float) -> str:
        """根据参与度评分划分等级。

        Args:
            score: 参与度评分（0~100）。

        Returns:
            参与度等级字符串。
        """
        if score >= 80:
            return "high"
        if score >= 60:
            return "normal"
        if score >= 40:
            return "distracted"
        if score >= 20:
            return "low"
        return "absent"

    @staticmethod
    def _classify_state(score: float) -> str:
        """根据参与度评分划分当前注意状态。

        Args:
            score: 参与度评分（0~100）。

        Returns:
            当前注意状态字符串。
        """
        if score >= 80:
            return "focusing"
        if score >= 60:
            return "attentive"
        if score >= 40:
            return "distracted"
        return "inattentive"

    @staticmethod
    def _compute_confidence(
        eye_state: Optional[EyeState],
        head_pose: Optional[HeadPoseResult],
        fatigue: Optional[FatigueResult],
    ) -> float:
        """计算参与度评估可信度。

        Args:
            eye_state: 眼部状态结果（可为 None）。
            head_pose: 头部姿态结果（可为 None）。
            fatigue: 疲劳评估结果（可为 None）。

        Returns:
            可信度值（0~1）。
        """
        confidences = []
        for obj in (eye_state, head_pose, fatigue):
            if obj is not None:
                confidences.append(float(obj.confidence))
        if not confidences:
            return 0.0
        return float(sum(confidences) / len(confidences))
