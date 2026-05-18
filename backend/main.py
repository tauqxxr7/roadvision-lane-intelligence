from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.pipeline.video_processor import VideoProcessor
from backend.utils.logger import configure_logging


configure_logging()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

processor = VideoProcessor(output_dir="runs")

app = FastAPI(
    title="RoadVision Lane Intelligence API",
    description="FastAPI service for lane overlay generation, departure warnings, and runtime metrics.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "roadvision-lane-intelligence"}


@app.post("/api/v1/process-video")
async def process_video(file: Annotated[UploadFile, File(...)]) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Upload a video file: MP4, MOV, AVI, or MKV.")

    upload_path = UPLOAD_DIR / f"{Path(file.filename or 'road_video').stem}{suffix}"
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        upload_path.write_bytes(content)

        metrics = processor.process_video(upload_path)
        return {
            "job_id": metrics.job_id,
            "metrics": metrics.__dict__,
            "download_url": f"/api/v1/results/{metrics.job_id}/video",
            "metrics_url": f"/api/v1/results/{metrics.job_id}/metrics",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("video_processing_failed filename=%s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/results/{job_id}/video")
def download_video(job_id: str) -> FileResponse:
    output_path = Path("runs") / job_id / "roadvision_lane_overlay.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found.")
    return FileResponse(output_path, media_type="video/mp4", filename=f"roadvision_{job_id}.mp4")


@app.get("/api/v1/results/{job_id}/metrics")
def download_metrics(job_id: str) -> FileResponse:
    metrics_path = Path("runs") / job_id / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    return FileResponse(metrics_path, media_type="application/json", filename=f"roadvision_{job_id}_metrics.json")
