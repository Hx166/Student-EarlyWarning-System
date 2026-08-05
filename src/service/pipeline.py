"""src/service/pipeline.py —— 主流程编排服务。

将摄像头采集、视觉分析、疲劳与参与度评估、
数据存储串联，按照设计文档定义的固定数据流顺序执行。

流程：
    Camera → CV(Fatigue+Engagement) → SQLite(FrameRecord/StateRecord/WarningLog)

其中 FrameRecord 存储每帧原始数据，
StateRecord 每秒聚合一次，WarningLog 存储风险预警。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from src.camera import Camera
from src.core.models import Frame
from src.config import get_database
from src.cv import CVProcessor
from src.database import DatabaseService
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 秒级聚合窗口（秒）
_AGGREGATION_WINDOW = 1.0
# 同类型预警最小间隔（秒），防止刷屏
_WARNING_COOLDOWN = 10.0
# 风险触发阈值
_WARNING_FATIGUE_HIGH = 80.0
_WARNING_FATIGUE_MEDIUM = 60.0
_WARNING_ENGAGEMENT_LOW = 40.0


class PipelineService:
    """系统主流程服务。

    Attributes:
        camera: 摄像头采集器。
        cv: 视觉处理流水线。
        db: 数据库服务。
        student_id: 当前学生 ID。
        session_id: 当前会话 ID。
        frame_record_enabled: 是否写入 FrameRecord。
    """

    def __init__(
        self,
        camera: Camera,
        cv: CVProcessor,
        db: DatabaseService,
        student_id: int,
        session_id: int,
    ) -> None:
        """初始化主流程服务。

        Args:
            camera: 摄像头采集器。
            cv: 视觉处理流水线。
            db: 数据库服务。
            student_id: 学生 ID。
            session_id: 会话 ID。
        """
        self.camera = camera
        self.cv = cv
        self.db = db
        self.student_id = student_id
        self.session_id = session_id
        # 读取配置中的 FrameRecord 开关，遵循 database.yaml.frame_record.enabled
        db_cfg = get_database().get("frame_record", {})
        self.frame_record_enabled = bool(db_cfg.get("enabled", True))
        # 秒级聚合缓冲
        self._agg_start: Optional[float] = None
        self._agg_ear = 0.0
        self._agg_pitch = 0.0
        self._agg_yaw = 0.0
        self._agg_roll = 0.0
        self._agg_perclos = 0.0
        self._agg_attention = 0.0
        self._agg_fatigue = 0.0
        self._agg_count = 0
        # 预警去重：同类型最近触发时刻
        self._last_warning_at: dict = {}

    def process_one(self) -> Optional[Frame]:
        """执行单帧采集与处理。

        读取一帧并进行视觉分析，
        将原始帧数据写入 FrameRecord，
        每秒聚合写入 StateRecord，并判断是否触发预警。

        Returns:
            处理后的帧对象；无帧时返回 None。
        """
        frame = self.camera.read()
        if frame is None:
            return None

        cv_result = self.cv.process(frame.image)

        self._persist_frame(cv_result)
        self._accumulate(cv_result)
        self._flush_if_window_elapsed()
        self._maybe_write_warning(cv_result)

        logger.debug(
            "帧 #%s 处理完成，参与度=%s，疲劳=%s",
            frame.frame_id,
            cv_result.engagement.engagement_score,
            cv_result.fatigue.fatigue_score,
        )
        return frame

    def _persist_frame(self, cv_result) -> None:
        """将当前帧的原始视觉数据写入 FrameRecord。

        Args:
            cv_result: 视觉处理结果。
        """
        if not self.frame_record_enabled:
            return
        self.db.insert_frame_record(
            student_id=self.student_id,
            session_id=self.session_id,
            ear=cv_result.eye.average_ear,
            mar=0.0,
            pitch=cv_result.head_pose.pitch,
            yaw=cv_result.head_pose.yaw,
            roll=cv_result.head_pose.roll,
        )

    def _accumulate(self, cv_result) -> None:
        """累积当前帧到秒级聚合缓冲。

        Args:
            cv_result: 视觉处理结果。
        """
        now = time.monotonic()
        if self._agg_start is None:
            self._agg_start = now
        self._agg_ear += cv_result.eye.average_ear
        self._agg_pitch += cv_result.head_pose.pitch
        self._agg_yaw += cv_result.head_pose.yaw
        self._agg_roll += cv_result.head_pose.roll
        self._agg_perclos += cv_result.eye.perclos
        self._agg_attention += cv_result.engagement.engagement_score
        self._agg_fatigue += cv_result.fatigue.fatigue_score
        self._agg_count += 1

    def _flush_if_window_elapsed(self) -> None:
        """当聚合窗口到期时写入 StateRecord。"""
        if self._agg_start is None or self._agg_count == 0:
            return
        now = time.monotonic()
        if now - self._agg_start < _AGGREGATION_WINDOW:
            return
        count = self._agg_count
        self.db.insert_state_record(
            student_id=self.student_id,
            session_id=self.session_id,
            avg_ear=self._agg_ear / count,
            avg_mar=0.0,
            avg_pitch=self._agg_pitch / count,
            avg_yaw=self._agg_yaw / count,
            avg_roll=self._agg_roll / count,
            perclos=self._agg_perclos / count,
            attention_score=self._agg_attention / count,
            fatigue_score=self._agg_fatigue / count,
        )
        logger.debug("已写入 StateRecord，聚合 %s 帧", count)
        self._reset_accumulator()

    def _reset_accumulator(self) -> None:
        """重置秒级聚合缓冲。"""
        self._agg_start = None
        self._agg_ear = 0.0
        self._agg_pitch = 0.0
        self._agg_yaw = 0.0
        self._agg_roll = 0.0
        self._agg_perclos = 0.0
        self._agg_attention = 0.0
        self._agg_fatigue = 0.0
        self._agg_count = 0

    def _maybe_write_warning(self, cv_result) -> None:
        """根据疲劳与参与度评分判断是否触发预警。

        Args:
            cv_result: 视觉处理结果。
        """
        fatigue_score = cv_result.fatigue.fatigue_score
        engagement_score = cv_result.engagement.engagement_score
        faces = cv_result.faces

        if fatigue_score >= _WARNING_FATIGUE_HIGH:
            self._write_warning(
                warning_type="fatigue",
                warning_level="high",
                reason="high_fatigue_score",
                attention_score=engagement_score,
            )
        elif fatigue_score >= _WARNING_FATIGUE_MEDIUM:
            self._write_warning(
                warning_type="fatigue",
                warning_level="medium",
                reason="moderate_fatigue_score",
                attention_score=engagement_score,
            )

        if engagement_score <= _WARNING_ENGAGEMENT_LOW:
            self._write_warning(
                warning_type="low_engagement",
                warning_level="medium",
                reason="low_engagement_score",
                attention_score=engagement_score,
            )

        # 长时间无人脸视为缺勤提示
        if not faces or faces is None:
            self._write_warning(
                warning_type="absent",
                warning_level="low",
                reason="face_not_present",
                attention_score=engagement_score,
            )

    def _write_warning(
        self,
        warning_type: str,
        warning_level: str,
        reason: str,
        attention_score: float,
    ) -> None:
        """写入预警记录，并进行同类型去重。

        Args:
            warning_type: 预警类型。
            warning_level: 预警等级。
            reason: 预警原因。
            attention_score: 专注度评分。
        """
        now = time.monotonic()
        last = self._last_warning_at.get(warning_type)
        # ???????????????????????????
        if last is not None and now - last < _WARNING_COOLDOWN:
            return
        self._last_warning_at[warning_type] = now
        self.db.insert_warning(
            student_id=self.student_id,
            session_id=self.session_id,
            warning_level=warning_level,
            warning_type=warning_type,
            warning_reason=reason,
            attention_score=attention_score,
        )
        logger.warning("触发预警 types=%s level=%s reason=%s",
                       warning_type, warning_level, reason)

    def run(self, max_frames: Optional[int] = None) -> int:
        """运行主循环。

        Args:
            max_frames: 最大处理帧数，None 表示无限运行。

        Returns:
            实际处理的帧数。
        """
        logger.info("主流程开始运行")
        processed = 0
        while max_frames is None or processed < max_frames:
            try:
                frame = self.process_one()
            except Exception as exc:  # 单帧异常不影响整体运行
                logger.error("处理单帧异常: %s", exc)
                continue
            if frame is None:
                # ?????????????????????
                break
            processed += 1
        logger.info("主流程结束，共处理 %s 帧", processed)
        return processed

    def close(self) -> None:
        """释放所有资源。"""
        self.cv.close()
        self.camera.close()
        logger.info("主流程资源已释放")
