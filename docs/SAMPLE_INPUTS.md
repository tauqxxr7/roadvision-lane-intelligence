# Sample Input Guide

RoadVision works best with short, forward-facing road footage where lane markings are visible in the lower half of the frame. This guide exists so reviewers can reproduce a clean first run without guessing what kind of video the pipeline expects.

## Recommended First Clip

- Format: MP4 with H.264 encoding
- Resolution: 1280x720 or 1920x1080
- Length: 10 to 30 seconds
- Camera angle: windshield or dashboard center view
- Lighting: daylight or evenly lit road
- Road type: straight or mildly curved road with visible lane paint

## Where To Put Local Videos

Place local videos in:

```text
sample_data/
```

Large video files are intentionally ignored by git. The repository keeps instructions and placeholders, while each developer or reviewer supplies their own road footage locally.

## Good Test Cases

Use these clips to exercise the system:

- Centered driving on a straight road
- Slight left drift and recovery
- Slight right drift and recovery
- Dashed white lane markings
- Solid lane markings
- Mild shadows across the lane

## Avoid For The First Demo

These clips are useful later, but they can make a first reviewer run look worse than the pipeline actually is:

- Night footage with glare
- Heavy rain or windshield wiper motion
- Strong shadows from trees or flyovers
- Lane markings mostly hidden by traffic
- Sharp curves where straight-line fitting is not enough
- Construction zones with temporary paint or cones

## Example Run

Start the API:

```bash
uvicorn backend.main:app --reload
```

Upload a video:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/process-video" \
  -F "file=@sample_data/road_clip.mp4"
```

Or use the dashboard:

```bash
streamlit run frontend/app.py
```

The processed video and metrics JSON will be written under `runs/{job_id}/`.
