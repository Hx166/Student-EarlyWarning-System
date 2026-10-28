"""src/service/offline_video_processor.py —— 真实视频离线特征提取（训练数据 PoC）。

将现有 CVPipeline 逐帧复用到离线视频：

    视频 → CVProcessor + OfflineFatigueAnalyzer
         → 秒级 StateRecord / BehaviorRecord
         → FeatureBuilder（30 秒窗口） → 训练特征 CSV

本模块只新增离线训练数据生成能力，不改动任何生产模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.models import BehaviorResult
from src.cv import CVProcessor
from src.cv.behavior.offline_fatigue import OfflineFatigueAnalyzer
from src.cv.behavior.clock import VideoClock
from src.models.features.builder import FeatureBuilder
from src.service.behavior_tracker import BehaviorEventTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 秒级聚合窗口（秒），与 PipelineService 保持一致
_AGG_WINDOW_SECONDS = 1.0


@dataclass
class OfflineVideoStats:
    """单视频离线处理统计摘要。

    Attributes:
        video_id: 视频标识。
        source_dataset: 来源数据集。
        label: 标签（字符串）。
        fps: 视频帧率。
        width / height: 视频分辨率。
        frame_count: 已读帧数。
        face_found: 检测到人脸的帧数。
        face_found_ratio: 人脸检出率（0~1）。
        face_lost_duration: 无人脸累计时长（视频秒）。
        face_lost_ratio: 无人脸时长占比（0~1）。
        processed_seconds: 实际处理的视频秒数。
        state_record_count: 生成的 StateRecord 条数。
        blink_total: 眨眼事件总数。
        head_down_total: 低头事件总数。
        yawn_total: 打哈欠事件总数。
        metrics: 逐秒统计指标（供报告汇总）。
        per_frame: 逐帧关键指标摘要（面部在场、EAR、姿态角）。
    """

    video_id: str = ""
    source_dataset: str = ""
    label: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    face_found: int = 0
    face_found_ratio: float = 0.0
    face_lost_duration: float = 0.0
    face_lost_ratio: float = 0.0
    processed_seconds: float = 0.0
    state_record_count: int = 0
    blink_total: int = 0
    head_down_total: int = 0
    yawn_total: int = 0
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    per_frame: List[Dict[str, Any]] = field(default_factory=list)


class OfflineVideoProcessor:
    """离线视频 → 训练特征处理器。

    Attributes:
        window_seconds: 特征窗口长度（秒）。
        output_dir: 输出目录根（records/features/reports）。
        max_frames_per_video: 单视频最大处理帧数（None 表示全部）。
        frame_stride: 抽帧步长（1 表示逐帧）。
    """

    def __init__(
        self,
        output_dir: str = "datasets/poc_output",
        window_seconds: int = 30,
        max_frames_per_video: Optional[int] = None,
        frame_stride: int = 1,
    ) -> None:
        """初始化离线视频处理器。

        Args:
            output_dir: 输出目录根。
            window_seconds: 特征窗口长度（秒）。
            max_frames_per_video: 单视频最大处理帧数。
            frame_stride: 抽帧步长（1 表示逐帧）。
        """
        self.output_dir = Path(output_dir)
        self.records_dir = self.output_dir / "records"
        self.features_dir = self.output_dir / "features"
        self.reports_dir = self.output_dir / "reports"
        for sub in (self.records_dir, self.features_dir, self.reports_dir):
            sub.mkdir(parents=True, exist_ok=True)
        self.window_seconds = window_seconds
        self.max_frames_per_video = max_frames_per_video
        self.frame_stride = max(1, int(frame_stride))
        self._builder = FeatureBuilder(window_seconds=window_seconds)
        logger.info(
            "离线视频处理器初始化：输出=%s 窗口=%s秒 步长=%s",
            self.output_dir, window_seconds, self.frame_stride,
        )

    # ------------------------------------------------------------------ #
    # 视频读取与时间轴
    # ------------------------------------------------------------------ #
    def read_video_info(self, video_path: str) -> Dict[str, Any]:
        """读取视频基本参数。

        Args:
            video_path: 视频文件路径。

        Returns:
            视频参数字典（fps/width/height/frame_count/duration）。
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0.0:
                fps = 30.0
            duration = frame_count / fps if frame_count > 0 else 0.0
            return {
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "duration": duration,
            }
        finally:
            cap.release()

    def video_clock(self, video_info: Dict[str, Any], frame_stride: int = 1) -> VideoClock:
        """按视频参数构造视频时间轴时钟。

        步长 s 抽帧时，第 k 帧（步长帧中的序号）对应视频时间
        k × stride / fps。

        Args:
            video_info: read_video_info 返回的参数字典。
            frame_stride: 抽帧步长。

        Returns:
            视频时间轴时钟。
        """
        fps = float(video_info.get("fps") or 30.0)
        return VideoClock.from_fps(fps / max(1, frame_stride))

    # ------------------------------------------------------------------ #
    # 单视频处理
    # ------------------------------------------------------------------ #
    def process_video(
        self,
        video_path: str,
        video_id: str,
        source_dataset: str,
        label: str = "",
        subject_id: str = "",
        cv_processor: Optional[CVProcessor] = None,
        frame_stride: int = 1,
    ) -> OfflineVideoStats:
        """处理单个视频并生成记录与特征 CSV。

        Args:
            video_path: 视频文件路径。
            video_id: 视频标识（写入元数据）。
            source_dataset: 来源数据集（写入元数据，不作为特征）。
            label: 训练标签（字符串）。
            subject_id: 被试标识（写入元数据，不作为特征）。
            cv_processor: 视觉处理器；为空时新建。
            frame_stride: 抽帧步长（1 表示逐帧）。

        Returns:
            视频处理统计摘要。

        Raises:
            FileNotFoundError: 视频文件不存在。
        """
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        import cv2

        own_cv = cv_processor is None
        cv_proc = cv_processor if cv_processor is not None else CVProcessor()
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {video_path}")
            info = self.read_video_info(video_path)
            clock = self.video_clock(info, frame_stride)
            # 疲劳分析使用独立的视频时间轴时钟，避免与主循环共享时钟
            # 导致每帧时间戳被推进两次（主循环 + fatigue 内部消费）
            fatigue_clock = self.video_clock(info, frame_stride)
            fatigue = OfflineFatigueAnalyzer(clock=fatigue_clock)
            tracker = BehaviorEventTracker()

            state_records: List[Dict[str, Any]] = []
            behavior_records: List[Dict[str, Any]] = []
            per_frame: List[Dict[str, Any]] = []
            aggr: Dict[str, float] = {}
            aggr_count = 0
            agg_second = 0

            seen = 0
            processed_frames = 0
            zero_time = datetime(2026, 1, 1, 0, 0, 0)
            face_lost_frames = 0
            blink_total = 0
            head_down_total = 0
            yawn_total = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if seen % self.frame_stride != 0:
                    seen += 1
                    continue
                seen += 1
                if self.max_frames_per_video is not None and processed_frames >= self.max_frames_per_video:
                    break

                video_ts = clock.monotonic()
                # STATE 阶段使用视频帧率作为感知刻度，见下方“疲劳按视频时间”
                # 这里直接以 videos 时间戳驱动疲劳窗口。
                # 先执行 CVProcessor（内部替换 fatigue 为离线适配器）
                self._patch_cv_fatigue(cv_proc, fatigue)
                cv_result = cv_proc.process(frame)

                if cv_result.faces:
                    face_lost_interval = 0.0
                else:
                    face_lost_interval = 1.0 / max(info.get("fps") or 30.0, 1e-6)
                    face_lost_frames += 1
                per_frame.append(
                    {
                        "frame": processed_frames,
                        "video_time": round(video_ts, 3),
                        "face_present": bool(cv_result.faces),
                        "ear": cv_result.eye.average_ear,
                        "pitch": cv_result.head_pose.pitch,
                        "yaw": cv_result.head_pose.yaw,
                        "roll": cv_result.head_pose.roll,
                        "perclos": cv_result.eye.perclos,
                        "fatigue": cv_result.fatigue.fatigue_score,
                        "attention": cv_result.engagement.engagement_score,
                        "blink": cv_result.eye.blink,
                    }
                )
                processed_frames += 1

                # 秒级聚合（按视频时间）
                for key in ("avg_ear", "avg_mar", "avg_pitch", "avg_yaw", "avg_roll",
                            "perclos", "attention_score", "fatigue_score"):
                    aggr[key] = aggr.get(key, 0.0) + getattr(
                        getattr(cv_result, _STATE_ATTR_MAP[key][0]), _STATE_ATTR_MAP[key][1]
                    )
                aggr_count += 1
                target_second = int(video_ts + 1e-9)
                while target_second > agg_second:
                    agg_second += 1
                    self._flush_state(
                        state_records, zero_time, agg_second - 1, aggr, aggr_count,
                        video_id, source_dataset, subject_id,
                    )
                    aggr = {}
                    aggr_count = 0
                    if target_second == agg_second:
                        break

                # 行为事件（按视频时间）
                now_dt = zero_time + timedelta(seconds=int(video_ts))
                events = tracker.update(cv_result, now_dt)
                for ev in events:
                    behavior_records.append(self._behavior_to_dict(ev, video_id, source_dataset, subject_id))
                    if ev.behavior_type == "blink":
                        blink_total += 1
                    elif ev.behavior_type == "head_down":
                        head_down_total += 1
                    elif ev.behavior_type == "yawn":
                        yawn_total += 1

            # flush 末尾聚合与未结束行为
            if aggr_count > 0:
                self._flush_state(
                    state_records, zero_time, agg_second, aggr, aggr_count,
                    video_id, source_dataset, subject_id,
                )
            for ev in tracker.flush(zero_time + timedelta(seconds=int(clock.monotonic()))):
                behavior_records.append(self._behavior_to_dict(ev, video_id, source_dataset, subject_id))
                if ev.behavior_type == "blink":
                    blink_total += 1
                elif ev.behavior_type == "head_down":
                    head_down_total += 1
                elif ev.behavior_type == "yawn":
                    yawn_total += 1

            processed_seconds = processed_frames / max(info.get("fps") or 30.0, 1e-6)
            stats = self._build_stats(
                video_id=video_id,
                source_dataset=source_dataset,
                label=label,
                info=info,
                processed_frames=processed_frames,
                face_lost_frames=face_lost_frames,
                processed_seconds=processed_seconds,
                state_records=state_records,
                blink_total=blink_total,
                head_down_total=head_down_total,
                yawn_total=yawn_total,
                per_frame=per_frame,
            )
            self._save_video_outputs(
                video_id=video_id,
                state_records=state_records,
                behavior_records=behavior_records,
                label=label,
                source_dataset=source_dataset,
                subject_id=subject_id,
                stats=stats,
            )
            return stats
        finally:
            if own_cv:
                cv_proc.close()

    # ------------------------------------------------------------------ #
    # 内部帮助
    # ------------------------------------------------------------------ #
    def _patch_cv_fatigue(self, cv_proc: CVProcessor, fatigue: OfflineFatigueAnalyzer) -> None:
        """将 CVProcessor 主脸疲劳分析器替换为离线适配器。

        Args:
            cv_proc: CVProcessor 实例。
            fatigue: 离线疲劳分析适配器。
        """
        cv_proc.fatigue_analyzer = fatigue  # type: ignore[assignment]

    def _flush_state(
        self,
        state_records: List[Dict[str, Any]],
        zero_time: datetime,
        idx: int,
        aggr: Dict[str, float],
        count: int,
        video_id: str,
        source_dataset: str,
        subject_id: str,
    ) -> None:
        """将秒级聚合结果写入状态记录列表。

        Args:
            state_records: 状态记录列表（追加目标）。
            zero_time: 时间起点（用于构造 record_time）。
            idx: 当前已写记录数（用于秒序号）。
            aggr: 聚合累加值。
            count: 聚合帧数。
            video_id / source_dataset / subject_id: 元数据。
        """
        record_time = zero_time + timedelta(seconds=idx + 1)
        state_records.append(
            {
                "state_id": idx + 1,
                "student_id": subject_id or video_id,
                "session_id": source_dataset,
                "record_time": record_time.isoformat(),
                "avg_ear": aggr.get("avg_ear", 0.0) / count,
                "avg_mar": aggr.get("avg_mar", 0.0) / count,
                "avg_pitch": aggr.get("avg_pitch", 0.0) / count,
                "avg_yaw": aggr.get("avg_yaw", 0.0) / count,
                "avg_roll": aggr.get("avg_roll", 0.0) / count,
                "perclos": aggr.get("perclos", 0.0) / count,
                "attention_score": aggr.get("attention_score", 0.0) / count,
                "fatigue_score": aggr.get("fatigue_score", 0.0) / count,
            }
        )

    def _behavior_to_dict(
        self, ev: BehaviorResult, video_id: str, source_dataset: str, subject_id: str
    ) -> Dict[str, Any]:
        """将行为事件转换为训练记录字典。

        Args:
            ev: 行为事件。
            video_id / source_dataset / subject_id: 元数据。

        Returns:
            行为记录字典。
        """
        return {
            "behavior_id": 0,
            "student_id": subject_id or video_id,
            "session_id": source_dataset,
            "behavior_type": ev.behavior_type,
            "start_time": ev.start_time.isoformat() if ev.start_time else None,
            "end_time": ev.end_time.isoformat() if ev.end_time else None,
            "duration": ev.duration,
        }

    def _build_stats(
        self,
        video_id: str,
        source_dataset: str,
        label: str,
        info: Dict[str, Any],
        processed_frames: int,
        face_lost_frames: int,
        processed_seconds: float,
        state_records: List[Dict[str, Any]],
        blink_total: int,
        head_down_total: int,
        yawn_total: int,
        per_frame: List[Dict[str, Any]],
    ) -> OfflineVideoStats:
        """汇总单视频统计信息。

        Args:
            各参数为单视频处理结果汇总。

        Returns:
            视频统计对象。
        """
        face_found = processed_frames - face_lost_frames
        face_found_ratio = face_found / processed_frames if processed_frames else 0.0
        face_lost_duration = face_lost_frames / max(info.get("fps") or 30.0, 1e-6)
        frame_seconds = 1.0 / max(info.get("fps") or 30.0, 1e-6)
        metrics = self._aggregate_metrics(per_frame, frame_seconds)
        return OfflineVideoStats(
            video_id=video_id,
            source_dataset=source_dataset,
            label=label,
            fps=float(info.get("fps") or 0.0),
            width=int(info.get("width") or 0),
            height=int(info.get("height") or 0),
            frame_count=processed_frames,
            face_found=int(face_found),
            face_found_ratio=round(face_found_ratio, 4),
            face_lost_duration=round(face_lost_duration, 3),
            face_lost_ratio=round(face_lost_frames / processed_frames if processed_frames else 0.0, 4),
            processed_seconds=round(processed_seconds, 2),
            state_record_count=len(state_records),
            blink_total=blink_total,
            head_down_total=head_down_total,
            yawn_total=yawn_total,
            metrics=metrics,
            per_frame=per_frame,
        )

    def _aggregate_metrics(
        self, per_frame: List[Dict[str, Any]], frame_seconds: float
    ) -> List[Dict[str, Any]]:
        """将逐帧指标聚合为秒级指标。

        Args:
            per_frame: 逐帧指标列表。
            frame_seconds: 每帧对应视频秒数。

        Returns:
            秒级指标列表。
        """
        if not per_frame:
            return []
        metrics: List[Dict[str, Any]] = []
        current_second = None
        bucket = None
        for row in per_frame:
            sec = int(row["video_time"])
            if current_second != sec:
                if bucket is not None:
                    metrics.append(self._close_bucket(bucket, frame_seconds))
                current_second = sec
                bucket = self._new_bucket(sec)
            bucket["count"] += 1
            bucket["ear_sum"] += row["ear"]
            bucket["perclos_sum"] += row["perclos"]
            bucket["pitch_sum"] += row["pitch"]
            bucket["yaw_sum"] += row["yaw"]
            bucket["roll_sum"] += row["roll"]
            bucket["fatigue_sum"] += row["fatigue"]
            bucket["attention_sum"] += row["attention"]
            if row["face_present"]:
                bucket["face_frames"] += 1
            if row["blink"]:
                bucket["blink_frames"] += 1
        if bucket is not None:
            metrics.append(self._close_bucket(bucket, frame_seconds))
        return metrics

    def _new_bucket(self, sec: int) -> Dict[str, Any]:
        """创建秒级统计桶。

        Args:
            sec: 秒序号。

        Returns:
            空统计桶字典。
        """
        return {
            "second": sec,
            "count": 0,
            "ear_sum": 0.0,
            "perclos_sum": 0.0,
            "pitch_sum": 0.0,
            "yaw_sum": 0.0,
            "roll_sum": 0.0,
            "fatigue_sum": 0.0,
            "attention_sum": 0.0,
            "face_frames": 0,
            "blink_frames": 0,
        }

    def _close_bucket(self, bucket: Dict[str, Any], frame_seconds: float) -> Dict[str, Any]:
        """关闭秒级统计桶。

        Args:
            bucket: 统计桶。
            frame_seconds: 每帧秒数。

        Returns:
            统计行字典。
        """
        count = bucket["count"]
        face_ratio = bucket["face_frames"] / count if count else 0.0
        return {
            "second": bucket["second"],
            "record_count": count,
            "face_ratio": round(face_ratio, 4),
            "ear_mean": round(bucket["ear_sum"] / count, 4) if count else 0.0,
            "perclos_mean": round(bucket["perclos_sum"] / count, 4) if count else 0.0,
            "pitch_mean": round(bucket["pitch_sum"] / count, 2) if count else 0.0,
            "yaw_mean": round(bucket["yaw_sum"] / count, 2) if count else 0.0,
            "roll_mean": round(bucket["roll_sum"] / count, 2) if count else 0.0,
            "fatigue_score_mean": round(bucket["fatigue_sum"] / count, 2) if count else 0.0,
            "attention_score_mean": round(bucket["attention_sum"] / count, 2) if count else 0.0,
            "blink_frames": bucket["blink_frames"],
        }

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #
    def _save_video_outputs(
        self,
        video_id: str,
        state_records: List[Dict[str, Any]],
        behavior_records: List[Dict[str, Any]],
        label: str,
        source_dataset: str,
        subject_id: str,
        stats: OfflineVideoStats,
    ) -> None:
        """保存单视频记录与特征 CSV。

        Args:
            video_id / state_records / behavior_records / label /
            source_dataset / subject_id / stats: 输出数据。
        """
        import pandas as pd

        safe_id = self._safe_name(video_id)
        records_path = self.records_dir / f"{safe_id}_state.csv"
        pd.DataFrame(state_records).to_csv(records_path, index=False, encoding="utf-8")

        if behavior_records:
            behavior_path = self.records_dir / f"{safe_id}_behavior.csv"
            pd.DataFrame(behavior_records).to_csv(behavior_path, index=False, encoding="utf-8")

        features = self._builder.build(state_records, behavior_records, labels=None)
        if not features.empty:
            if len(state_records) == len(features):
                features.insert(0, "record_time", [r["record_time"] for r in state_records])
            features.insert(0, "source_dataset", source_dataset)
            features.insert(1, "subject_id", subject_id or video_id)
            features.insert(2, "video_id", video_id)
            features.insert(3, "label", label)
            features_path = self.features_dir / f"{safe_id}_features.csv"
            features.to_csv(features_path, index=False, encoding="utf-8")

        self._save_report(video_id, stats)

    def _save_report(self, video_id: str, stats: OfflineVideoStats) -> None:
        """保存单视频统计报告 JSON。

        Args:
            video_id: 视频标识。
            stats: 视频统计对象。
        """
        import json

        report_path = self.reports_dir / f"{self._safe_name(video_id)}_stats.json"
        report_path.write_text(
            json.dumps(stats.__dict__, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _safe_name(name: str) -> str:
        """将视频标识转换为安全文件名。

        Args:
            name: 原始视频标识。

        Returns:
            安全文件名。
        """
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))

    @staticmethod
    def _STATE_ATTR_MAP():
        """状态字段到 CVResult 属性路径的映射。"""
        return {
            "avg_ear": ("eye", "average_ear"),
            "avg_mar": ("mouth", "mar"),
            "avg_pitch": ("head_pose", "pitch"),
            "avg_yaw": ("head_pose", "yaw"),
            "avg_roll": ("head_pose", "roll"),
            "perclos": ("eye", "perclos"),
            "attention_score": ("engagement", "engagement_score"),
            "fatigue_score": ("fatigue", "fatigue_score"),
        }


_STATE_ATTR_MAP = OfflineVideoProcessor._STATE_ATTR_MAP()

__all__ = ["OfflineVideoStats", "OfflineVideoProcessor"]
