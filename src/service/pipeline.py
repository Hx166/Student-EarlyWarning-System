"""src/service/pipeline.py —— 主流程编排服务。

将摄像头采集、视觉分析、行为评估与数据存储串联，
按照设计文档定义的固定数据流顺序执行。
"""

from __future__ import annotations

from typing import Optional

from src.camera import Camera
from src.core.models import Frame
from src.cv import CVProcessor
from src.database import DatabaseService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineService:
    """系统主流程服务。

    Attributes:
        camera: 摄像头采集器。
        cv: 视觉处理流水线。
        db: 数据库服务。
        student_id: 当前学生 ID。
        session_id: 当前会话 ID。
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

    def process_one(self) -> Optional[Frame]:
        """执行单帧采集与处理。

        读取一帧画面并进行视觉分析，
        将每秒状态统计数据写入数据库。

        Returns:
            处理后的帧对象；无帧时返回 None。
        """
        frame = self.camera.read()
        if frame is None:
            return None

        cv_result = self.cv.process(frame.image)

        # 将视觉特征写入数据库（此处为演示聚合，按帧写入状态表）
        self.db.insert_state_record(
            student_id=self.student_id,
            session_id=self.session_id,
            avg_ear=cv_result.eye.average_ear,
            avg_mar=0.0,
            avg_pitch=cv_result.head_pose.pitch,
            avg_yaw=cv_result.head_pose.yaw,
            avg_roll=cv_result.head_pose.roll,
            perclos=cv_result.eye.perclos,
            attention_score=cv_result.engagement.engagement_score,
            fatigue_score=cv_result.fatigue.fatigue_score,
        )
        logger.debug(
            "帧 #%s 处理完成，参与度=%s，疲劳=%s",
            frame.frame_id,
            cv_result.engagement.engagement_score,
            cv_result.fatigue.fatigue_score,
        )
        return frame

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
                continue
            processed += 1
        logger.info("主流程结束，共处理 %s 帧", processed)
        return processed

    def close(self) -> None:
        """释放所有资源。"""
        self.cv.close()
        self.camera.close()
        logger.info("主流程资源已释放")
