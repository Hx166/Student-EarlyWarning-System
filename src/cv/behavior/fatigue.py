"""src/cv/behavior/fatigue.py —— 疲劳状态（Fatigue）评估模块。

融合眼部（EAR/PERCLOS/Blink）与头部姿态（HeadPose）特征
计算学生疲劳评分，并在时间窗口内维护眨眼率与低头时长统计。

评分依据 docs/04-CV模块/Fatigue.md 设计规范，
采用加权融合方法：
    fatigue_score = 0.50 × PERCLOS + 0.20 × BlinkRate
                    + 0.15 × HeadDown + 0.15 × Yawn
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple

from src.core.models import (
    EyeState,
    FatigueResult,
    HeadPoseResult,
    MouthState,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FatigueAnalyzer:
    """疲劳状态评估器。

    实时接收眼部与头部姿态数据，
    在滑动时间窗口内维护 PERCLOS、眨眼率与低头时长统计，
    并输出综合疲劳评分与等级。

    Attributes:
        fatigue_window: 疲劳统计时间窗口（秒）。
        perclos_weight: PERCLOS 组件权重。
        blink_weight: 眨眼率组件权重。
        head_down_weight: 低头时长组件权重。
        yawn_weight: 打哈欠组件权重。
    """

    def __init__(self) -> None:
        """初始化疲劳评估器。"""
        self.fatigue_window = 60.0
        self.perclos_weight = 0.50
        self.blink_weight = 0.20
        self.head_down_weight = 0.15
        self.yawn_weight = 0.15
        # 上一帧时间戳（monotonic）
        self._last_ts: Optional[float] = None
        # 窗口内眨眼事件时间戳（monotonic）
        self._blink_times: Deque[float] = deque()
        # 窗口内低头事件（开始时间戳, 持续时长秒）
        self._head_down_events: Deque[Tuple[float, float]] = deque()
        # 窗口内打哈欠起始时间戳（monotonic）
        self._yawn_times: Deque[float] = deque()
        # 窗口内打哈欠帧（时间戳, 持续时长秒）
        self._yawn_frames: Deque[Tuple[float, float]] = deque()
        # 上一帧是否处于打哈欠状态
        self._last_yawn: bool = False
        logger.info("疲劳评估器已初始化，窗口=%s秒", self.fatigue_window)

    def analyze(
        self,
        eye_state: Optional[EyeState] = None,
        head_pose: Optional[HeadPoseResult] = None,
        mouth_state: Optional[MouthState] = None,
    ) -> FatigueResult:
        """综合评估疲劳状态。

        Args:
            eye_state: 眼部状态结果（可为 None）。
            head_pose: 头部姿态结果（可为 None）。
            mouth_state: 嘴部状态结果（可为 None）。

        Returns:
            疲劳评估结果对象。
        """
        now = time.monotonic()
        self._advance_window(now)
        dt = self._frame_delta(now)

        # 更新窗口内事件统计
        if eye_state is not None and eye_state.blink:
            self._blink_times.append(now)

        is_head_down = self._is_head_down(head_pose)
        if is_head_down and dt > 0.0:
            self._head_down_events.append((now, dt))

        is_yawn = self._is_yawn(mouth_state)
        if is_yawn and not self._last_yawn:
            self._yawn_times.append(now)
        if is_yawn and dt > 0.0:
            self._yawn_frames.append((now, dt))
        self._last_yawn = is_yawn

        blink_rate = self._compute_blink_rate(now)
        head_down_duration = self._sum_head_down(now)
        yawn_rate = self._compute_yawn_rate(now)
        yawn_duration = self._sum_yawn(now)
        perclos = eye_state.perclos if eye_state is not None else 0.0

        fatigue_score = self._compute_score(
            perclos=perclos,
            blink_rate=blink_rate,
            head_down_duration=head_down_duration,
            yawn_duration=yawn_duration,
        )
        fatigue_level = self._classify_level(fatigue_score)
        confidence = self._compute_confidence(eye_state, head_pose, mouth_state)

        return FatigueResult(
            perclos=perclos,
            blink_rate=blink_rate,
            head_down_duration=head_down_duration,
            yawn_rate=yawn_rate,
            yawn_duration=yawn_duration,
            fatigue_score=fatigue_score,
            fatigue_level=fatigue_level,
            confidence=confidence,
        )

    def _frame_delta(self, now: float) -> float:
        """计算本帧与上帧的时间间隔。

        Args:
            now: 当前时间戳（monotonic）。

        Returns:
            时间间隔（秒）；首帧返回 0。
        """
        if self._last_ts is None:
            self._last_ts = now
            return 0.0
        dt = max(0.0, now - self._last_ts)
        self._last_ts = now
        return dt

    def _advance_window(self, now: float) -> None:
        """清除超出时间窗口的历史事件。

        Args:
            now: 当前时间戳（monotonic）。
        """
        cutoff = now - self.fatigue_window
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        while self._head_down_events and self._head_down_events[0][0] < cutoff:
            self._head_down_events.popleft()
        while self._yawn_times and self._yawn_times[0] < cutoff:
            self._yawn_times.popleft()
        while self._yawn_frames and self._yawn_frames[0][0] < cutoff:
            self._yawn_frames.popleft()

    @staticmethod
    def _is_head_down(head_pose: Optional[HeadPoseResult]) -> bool:
        """判断当前是否处于低头状态。

        Args:
            head_pose: 头部姿态结果（可为 None）。

        Returns:
            低头返回 True，否则 False。
        """
        if head_pose is None or head_pose.confidence <= 0.0:
            return False
        return head_pose.pose_status == "LookDown"

    def _compute_blink_rate(self, now: float) -> float:
        """计算窗口内的眨眼率（次/分钟）。

        Args:
            now: 当前时间戳（monotonic）。

        Returns:
            眨眼率（次/分钟）。
        """
        cutoff = now - self.fatigue_window
        live = [ts for ts in self._blink_times if ts >= cutoff]
        return float(len(live) * 60.0 / self.fatigue_window)

    def _sum_head_down(self, now: float) -> float:
        """求和窗口内的低头累计时长（秒）。

        Args:
            now: 当前时间戳（monotonic）。

        Returns:
            窗口内低头累计时长（秒）。
        """
        cutoff = now - self.fatigue_window
        return float(sum(dt for ts, dt in self._head_down_events if ts >= cutoff))

    @staticmethod
    def _is_yawn(mouth_state: Optional[MouthState]) -> bool:
        """判断当前是否处于打哈欠状态。

        Args:
            mouth_state: 嘴部状态结果（可为 None）。

        Returns:
            打哈欠返回 True，否则 False。
        """
        if mouth_state is None or mouth_state.confidence <= 0.0:
            return False
        return bool(mouth_state.is_yawn)

    def _compute_yawn_rate(self, now: float) -> float:
        """计算窗口内的打哈欠频率（次/分钟）。

        Args:
            now: 当前时间戳（monotonic）。

        Returns:
            打哈欠频率（次/分钟）。
        """
        cutoff = now - self.fatigue_window
        live = [ts for ts in self._yawn_times if ts >= cutoff]
        return float(len(live) * 60.0 / self.fatigue_window)

    def _sum_yawn(self, now: float) -> float:
        """求和窗口内的打哈欠累计时长（秒）。

        Args:
            now: 当前时间戳（monotonic）。

        Returns:
            窗口内打哈欠累计时长（秒）。
        """
        cutoff = now - self.fatigue_window
        return float(sum(dt for ts, dt in self._yawn_frames if ts >= cutoff))

    def _compute_score(
        self,
        perclos: float,
        blink_rate: float,
        head_down_duration: float,
        yawn_duration: float = 0.0,
    ) -> float:
        """计算综合疲劳评分（0~100）。

        Args:
            perclos: 闭眼比例（0~1）。
            blink_rate: 眨眼率（次/分钟）。
            head_down_duration: 低头累计时长（秒）。
            yawn_duration: 打哈欠累计时长（秒）。

        Returns:
            疲劳评分（0~100）。
        """
        perclos_part = min(max(perclos, 0.0), 1.0)
        # 眨眼率超过 30 次/分钟视为显著疲劳
        blink_part = min(max(blink_rate / 30.0, 0.0), 1.0)
        head_down_part = min(max(head_down_duration / self.fatigue_window, 0.0), 1.0)
        yawn_part = min(max(yawn_duration / self.fatigue_window, 0.0), 1.0)

        score = 100.0 * (
            self.perclos_weight * perclos_part
            + self.blink_weight * blink_part
            + self.head_down_weight * head_down_part
            + self.yawn_weight * yawn_part
        )
        return float(round(score, 2))

    @staticmethod
    def _classify_level(score: float) -> str:
        """根据疲劳评分划分等级。

        Args:
            score: 疲劳评分（0~100）。

        Returns:
            疲劳等级字符串。
        """
        if score < 30:
            return "normal"
        if score < 60:
            return "mild"
        if score < 80:
            return "moderate"
        return "severe"

    @staticmethod
    def _compute_confidence(
        eye_state: Optional[EyeState],
        head_pose: Optional[HeadPoseResult],
        mouth_state: Optional[MouthState] = None,
    ) -> float:
        """计算疲劳评估可信度。

        Args:
            eye_state: 眼部状态结果（可为 None）。
            head_pose: 头部姿态结果（可为 None）。
            mouth_state: 嘴部状态结果（可为 None）。

        Returns:
            可信度值（0~1）。
        """
        confidences = []
        if eye_state is not None:
            confidences.append(eye_state.confidence)
        if head_pose is not None:
            confidences.append(head_pose.confidence)
        if mouth_state is not None:
            confidences.append(mouth_state.confidence)
        if not confidences:
            return 0.0
        return float(sum(confidences) / len(confidences))
