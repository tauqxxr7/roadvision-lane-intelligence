from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.pipeline.video_processor import VideoProcessor
from backend.utils.logger import configure_logging


configure_logging()

st.set_page_config(
    page_title="RoadVision Lane Intelligence",
    page_icon="RV",
    layout="wide",
)


def render_metric(label: str, value: str, caption: str) -> None:
    st.metric(label=label, value=value, help=caption)


def main() -> None:
    st.title("RoadVision Lane Intelligence")
    st.caption("Real-time lane overlay, drift warnings, and performance metrics for road footage.")

    with st.sidebar:
        st.header("Run Controls")
        uploaded = st.file_uploader("Road footage", type=["mp4", "mov", "avi", "mkv"])
        st.info("Use a forward-facing driving clip with visible lane markings. Keep the first test under 30 seconds.")

    left, right = st.columns([0.58, 0.42], gap="large")

    with left:
        st.subheader("Input Footage")
        if uploaded:
            st.video(uploaded)
        else:
            st.empty().container().write("Upload a road video to start a lane intelligence run.")

    with right:
        st.subheader("System Trace")
        st.write(
            "The pipeline crops the drivable region, extracts stable lane edges, fits left/right lane boundaries, "
            "and compares the lane center with the camera center for departure warnings."
        )

    if not uploaded:
        render_empty_state()
        return

    if st.button("Process footage", type="primary", use_container_width=True):
        progress = st.progress(0, text="Preparing video processing job")
        status = st.status("Running lane detection", expanded=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            input_path = Path(tmp.name)

        def update_progress(done: int, total: int) -> None:
            if total > 0:
                progress.progress(min(done / total, 1.0), text=f"Processed {done} / {total} frames")

        processor = VideoProcessor(output_dir="runs")
        try:
            metrics = processor.process_video(input_path, progress_callback=update_progress)
        except Exception as exc:
            status.update(label="Processing failed", state="error")
            st.error(str(exc))
            return

        status.update(label="Processing complete", state="complete")
        progress.progress(1.0, text="Lane intelligence run complete")
        render_results(metrics)


def render_empty_state() -> None:
    st.divider()
    cols = st.columns(4)
    cols[0].metric("Pipeline", "OpenCV", "CPU-first classical CV")
    cols[1].metric("Output", "MP4", "Lane overlay + HUD")
    cols[2].metric("Signals", "Drift", "Left/right/centered status")
    cols[3].metric("Metrics", "FPS", "Latency and low-confidence frames")


def render_results(metrics) -> None:
    st.divider()
    st.subheader("Run Results")

    top = st.columns(5)
    render_pairs = [
        ("Resolution", f"{metrics.width}x{metrics.height}", "Input frame dimensions"),
        ("Processing FPS", f"{metrics.processing_fps:.2f}", "Measured end-to-end processing speed"),
        ("Avg Latency", f"{metrics.avg_latency_ms:.1f} ms", "Lane detector latency per frame"),
        ("Departure Frames", str(metrics.lane_departure_events), "Frames flagged as left/right drift"),
        ("Low Confidence", str(metrics.low_confidence_frames), "Frames where lane evidence was weak"),
    ]
    for col, (label, value, caption) in zip(top, render_pairs):
        with col:
            render_metric(label, value, caption)

    video_path = Path(metrics.output_path)
    st.video(str(video_path))

    metrics_json = json.dumps(metrics.__dict__, indent=2)
    st.download_button(
        "Download processed video",
        data=video_path.read_bytes(),
        file_name=f"roadvision_{metrics.job_id}.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

    st.download_button(
        "Download metrics JSON",
        data=metrics_json,
        file_name=f"roadvision_{metrics.job_id}_metrics.json",
        mime="application/json",
        use_container_width=True,
    )

    chart_data = pd.DataFrame(
        {
            "metric": ["processed frames", "departure frames", "low confidence frames"],
            "count": [metrics.processed_frames, metrics.lane_departure_events, metrics.low_confidence_frames],
        }
    )
    st.bar_chart(chart_data, x="metric", y="count", color="#2f6f73")


if __name__ == "__main__":
    main()
