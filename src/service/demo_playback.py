"""阶段 G：DAiSEE 片段回放、缓存与模型说明数据服务。"""

from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.models import ModelRegistry
from src.models.features.builder import FeatureBuilder
from src.models.features.definitions import FEATURE_VECTOR_COLUMNS
from src.models.labels import ENGAGEMENT_LABEL_MAPPING
from src.models.risk_prediction import RiskPredictor
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DemoPlaybackService:
    """管理演示片段清单、回放计算与 LRU 缓存。"""

    def __init__(
        self,
        sample_path: Optional[Path] = None,
        model_version: Optional[str] = None,
        cache_size: int = 20,
        clips: Optional[pd.DataFrame] = None,
        processor_factory: Optional[Callable[[str], Any]] = None,
        predictor: Optional[Any] = None,
    ) -> None:
        """初始化演示回放服务。

        Args:
            sample_path: 冻结片段清单；为空时使用阶段 C 默认路径。
            model_version: 模型版本；为空时读取系统配置。
            cache_size: LRU 缓存条目上限。
            clips: 测试注入的片段 DataFrame。
            processor_factory: 测试注入的视频处理器工厂。
            predictor: 测试注入的预测器。
        """
        training = load_config("training")["training"]
        prediction = load_config("system").get("prediction", {})
        datasets = Path(PROJECT_ROOT) / str(training.get("datasets_dir", "datasets"))
        self.sample_path = sample_path or datasets / "processed" / "stage_c_sample.csv"
        self.data_root = Path(training["daisee_root"]).resolve()
        self.model_version = model_version or prediction.get("model_version", "engagement-v1")
        self._processor_factory = processor_factory
        self._predictor = predictor
        self._clips = self._load_clips(clips)
        self._cache_size = cache_size
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _load_clips(self, clips: Optional[pd.DataFrame]) -> pd.DataFrame:
        """读取并校验冻结片段清单。"""
        if clips is not None:
            frame = clips.copy()
        elif self.sample_path.exists():
            frame = pd.read_csv(self.sample_path)
        else:
            logger.warning("演示片段清单不存在: %s", self.sample_path)
            return pd.DataFrame(columns=("clip_id", "engagement", "frames", "ext", "path"))
        required = {"clip_id", "engagement", "frames", "ext", "path"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"演示片段清单缺少列: {missing}")
        frame["clip_id"] = frame["clip_id"].astype(str)
        if frame["clip_id"].duplicated().any():
            raise ValueError("演示片段清单存在重复 clip_id")
        return frame.reset_index(drop=True)

    def get_clips(self) -> list[dict[str, Any]]:
        """返回回放页可用的 20 条片段。"""
        return [
            {
                "clip_id": str(row["clip_id"]),
                "engagement": int(row["engagement"]),
                "frames": int(row["frames"]),
                "ext": str(row["ext"]).lower(),
                "playable": str(row["ext"]).lower() == ".mp4",
            }
            for _, row in self._clips.iterrows()
        ]

    def _require_clip(self, clip_id: str) -> pd.Series:
        """只允许访问冻结清单内的 clip_id。"""
        matches = self._clips[self._clips["clip_id"] == str(clip_id)]
        if matches.empty:
            raise KeyError(f"片段不在演示清单中: {clip_id}")
        return matches.iloc[0]

    def video_path(self, clip_id: str) -> Path:
        """返回清单内 MP4 的受控路径。"""
        row = self._require_clip(clip_id)
        if str(row["ext"]).lower() != ".mp4":
            raise FileNotFoundError(f"片段不是 MP4: {clip_id}")
        path = Path(str(row["path"])).resolve()
        if self.data_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"视频文件不存在或越界: {clip_id}")
        return path

    def _get_predictor(self) -> Any:
        """延迟创建 Engagement 预测器。"""
        if self._predictor is None:
            self._predictor = RiskPredictor(
                registry=ModelRegistry(),
                model_name="xgboost",
                model_version=self.model_version,
            )
        return self._predictor

    def _processor(self, output_dir: str) -> Any:
        """创建视频处理器。"""
        if self._processor_factory is not None:
            return self._processor_factory(output_dir)
        from src.service.offline_video_processor import OfflineVideoProcessor

        return OfflineVideoProcessor(output_dir=output_dir, window_seconds=10)

    def _run_clip(self, row: pd.Series) -> dict[str, Any]:
        """完整处理一个片段并预测。"""
        started = time.perf_counter()
        clip_id = str(row["clip_id"])
        safe_id = "".join(
            char if char.isalnum() or char in ("-", "_") else "_" for char in clip_id
        )
        with tempfile.TemporaryDirectory(prefix=f"demo_{safe_id}_") as temporary:
            processor = self._processor(temporary)
            processor.process_video(
                video_path=str(row["path"]), video_id=clip_id,
                source_dataset="daisee", label=str(int(row["engagement"])),
                subject_id=str(row.get("subject_id", "")),
            )
            root = Path(temporary)
            state = pd.read_csv(root / "records" / f"{safe_id}_state.csv")
            behavior_path = root / "records" / f"{safe_id}_behavior.csv"
            behavior = (
                pd.read_csv(behavior_path)
                if behavior_path.exists()
                else pd.DataFrame(columns=("behavior_type", "duration"))
            )
            features = FeatureBuilder(window_seconds=10).build(
                state.to_dict("records"), behavior.to_dict("records")
            )
            result, probabilities = self._get_predictor().predict_with_proba(
                features.iloc[[-1]]
            )
        probability_map = {
            ENGAGEMENT_LABEL_MAPPING[index]: float(probabilities[index])
            for index in range(4)
        }
        series = [
            {
                "second": index + 1,
                "ear": float(record["avg_ear"]),
                "perclos": float(record["perclos"]),
                "pitch": float(record["avg_pitch"]),
                "yaw": float(record["avg_yaw"]),
                "fatigue": float(record["fatigue_score"]),
                "attention": float(record["attention_score"]),
            }
            for index, record in enumerate(state.to_dict("records"))
        ]
        return {
            "clip_id": clip_id,
            "true_level": ENGAGEMENT_LABEL_MAPPING[int(row["engagement"])],
            "pred_level": result.risk_label,
            "confidence": float(result.risk_probability),
            "probabilities": probability_map,
            "series": series,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }

    def run(self, clip_id: str) -> dict[str, Any]:
        """运行回放；同一片段第二次从内存缓存返回。"""
        row = self._require_clip(clip_id)
        with self._lock:
            if clip_id in self._cache:
                self._cache.move_to_end(clip_id)
                return copy.deepcopy(self._cache[clip_id])
            result = self._run_clip(row)
            self._cache[clip_id] = result
            self._cache.move_to_end(clip_id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return copy.deepcopy(result)
