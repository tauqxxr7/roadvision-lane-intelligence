from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2

from backend.pipeline.lane_detector import LaneDetector


logger = logging.getLogger(__name__)


ProgressCallback = Optional[Callable[[int, int], None]]


@dataclass
class VideoMetrics:
    job_id: str
    input_path: str
    output_path: str
    metrics_path: str
    frame_count: int
    processed_frames: int
    width: int
    height: int
    source_fps: float
    processing_fps: float
    avg_latency_ms: float
    max_latency_ms: float
    lane_departure_events: int
    low_confidence_frames: int


class VideoProcessor:
    def __init__(self, output_dir: str = "runs", detector: LaneDetector | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector = detector or LaneDetector()

    def process_video(self, input_path: str | Path, progress_callback: ProgressCallback = None) -> VideoMetrics:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the uploaded video. Try MP4/H.264 or MOV.")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("Video metadata is invalid; width and height must be positive.")

        job_id = uuid4().hex[:12]
        run_dir = self.output_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "roadvision_lane_overlay.mp4"
        metrics_path = run_dir / "metrics.json"

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )

        processed_frames = 0
        latencies_ms: list[float] = []
        departure_events = 0
        low_confidence_frames = 0
        started = time.perf_counter()

        logger.info("video_processing_started job_id=%s input=%s", job_id, input_path.name)

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_started = time.perf_counter()
                result = self.detector.process_frame(frame)
                latency_ms = (time.perf_counter() - frame_started) * 1000

                if result.departure_state in {"DRIFTING_LEFT", "DRIFTING_RIGHT"}:
                    departure_events += 1
                if result.confidence < 0.45:
                    low_confidence_frames += 1

                annotated = self._draw_hud(result.frame, result.departure_state, result.confidence, latency_ms)
                writer.write(annotated)

                processed_frames += 1
                latencies_ms.append(latency_ms)
                if progress_callback:
                    progress_callback(processed_frames, frame_count)
        finally:
            capture.release()
            writer.release()

        elapsed = max(time.perf_counter() - started, 1e-6)
        metrics = VideoMetrics(
            job_id=job_id,
            input_path=str(input_path),
            output_path=str(output_path),
            metrics_path=str(metrics_path),
            frame_count=frame_count,
            processed_frames=processed_frames,
            width=width,
            height=height,
            source_fps=round(source_fps, 2),
            processing_fps=round(processed_frames / elapsed, 2),
            avg_latency_ms=round(sum(latencies_ms) / max(len(latencies_ms), 1), 2),
            max_latency_ms=round(max(latencies_ms or [0.0]), 2),
            lane_departure_events=departure_events,
            low_confidence_frames=low_confidence_frames,
        )

        metrics_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
        logger.info(
            "video_processing_completed job_id=%s frames=%s fps=%s avg_latency_ms=%s",
            job_id,
            processed_frames,
            metrics.processing_fps,
            metrics.avg_latency_ms,
        )
        return metrics

    def _draw_hud(self, frame, state: str, confidence: float, latency_ms: float):
        color = (50, 210, 90) if state == "CENTERED" else (40, 140, 255)
        if state in {"DRIFTING_LEFT", "DRIFTING_RIGHT"}:
            color = (40, 40, 235)

        cv2.rectangle(frame, (18, 18), (430, 106), (12, 18, 24), -1)
        cv2.putText(frame, f"RoadVision: {state}", (34, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        cv2.putText(
            frame,
            f"confidence {confidence:.2f} | latency {latency_ms:.1f} ms",
            (34, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 236, 240),
            1,
        )
        return frame
