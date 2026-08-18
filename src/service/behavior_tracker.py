"""src/service/behavior_tracker.py —— 行为事件状态机。

将 CVProcessor 每帧输出的视觉状态转换为行为事件（BehaviorResult）：
- 连续型行为（head_down / yawn / looking_away / absent）按状态起止生成事件；
- 瞬时型行为（blink）按上升沿生成单次事件。

事件语义与 Fatigue / Engagement / EyeEAR 的内部状态保持一致。
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from src.core.models import BehaviorResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 连续型行为事件类型（需要跟踪开始与结束）
_CONTINUOUS_TYPES = ("head_down", "yawn", "looking_away", "absent")


class BehaviorEventTracker:
    """行为事件跟踪器。

    维护各连续型行为的活跃起始时间，并在状态切换时输出完成事件。

    Attributes:
        _active_start: 各连续型行为的活跃起始时间。
        _prev_blink: 上一帧是否发生眨眼（用于上升沿检测）。
    """

    def __init__(self) -> None:
        """初始化行为事件跟踪器。"""
        self._active_start: Dict[str, datetime] = {}
        self._prev_blink: bool = False
        logger.info("BehaviorEventTracker 已初始化")

    def update(self, cv_result, now: datetime) -> List[BehaviorResult]:
        """更新单帧状态并返回本帧新完成的行为事件。

        Args:
            cv_result: 单帧视觉处理结果（CVProcessResult）。
            now: 当前时间。

        Returns:
            本帧新完成的行为事件列表（可能为空）。
        """
        states = self._detect_states(cv_result)
        events: List[BehaviorResult] = []

        for btype in _CONTINUOUS_TYPES:
            active = states.get(btype, False)
            if active and btype not in self._active_start:
                self._active_start[btype] = now
            elif not active and btype in self._active_start:
                start = self._active_start.pop(btype)
                events.append(self._build_event(btype, start, now))

        blink = states.get("blink", False)
        if blink and not self._prev_blink:
            events.append(
                BehaviorResult(
                    behavior_type="blink",
                    start_time=now,
                    end_time=now,
                    duration=0.0,
                )
            )
        self._prev_blink = blink

        return events

    def flush(self, now: datetime) -> List[BehaviorResult]:
        """结束所有仍在活跃中的行为事件。

        Args:
            now: 结束时间。

        Returns:
            所有被结束的行为事件列表。
        """
        events: List[BehaviorResult] = []
        for btype, start in list(self._active_start.items()):
            events.append(self._build_event(btype, start, now))
        self._active_start.clear()
        return events

    def _detect_states(self, cv_result) -> Dict[str, bool]:
        """由单帧视觉结果判定各行为状态。

        Args:
            cv_result: 单帧视觉处理结果。

        Returns:
            行为状态字典。
        """
        faces = getattr(cv_result, "faces", None) or []
        face_present = bool(faces)
        pose = getattr(cv_result, "pose", None)
        pose_present = pose is not None and float(pose.confidence) > 0.0

        head_pose = getattr(cv_result, "head_pose", None)
        pose_status = getattr(head_pose, "pose_status", "Normal") if head_pose else "Normal"
        eye = getattr(cv_result, "eye", None)
        mouth = getattr(cv_result, "mouth", None)

        return {
            "blink": bool(eye.blink) if eye is not None else False,
            "head_down": pose_status == "LookDown",
            "yawn": bool(mouth.is_yawn) if mouth is not None else False,
            # 无人脸但仍有身体姿态 → 疑似转头/离开视线
            "looking_away": (
                pose_status in ("LookLeft", "LookRight", "Tilt")
                or (not face_present and pose_present)
            ),
            # 无人脸且无身体姿态 → 离开画面
            "absent": (not face_present) and (not pose_present),
        }

    @staticmethod
    def _build_event(
        btype: str, start: datetime, end: datetime
    ) -> BehaviorResult:
        """构造行为事件结果对象。

        Args:
            btype: 行为类型。
            start: 开始时间。
            end: 结束时间。

        Returns:
            行为事件结果对象。
        """
        duration = max((end - start).total_seconds(), 0.0)
        return BehaviorResult(
            behavior_type=btype,
            start_time=start,
            end_time=end,
            duration=duration,
        )
