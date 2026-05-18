# Engineering Notes

## Design Philosophy

RoadVision starts with a classical CV baseline because lane departure logic benefits from explainability. Every stage has a visible reason: remove noise, focus on the road, identify lane-like edges, fit plausible boundaries, then derive a vehicle offset.

The goal is not to claim production autonomy. The goal is to show an honest, measurable perception system with clean engineering boundaries.

## Pipeline Decisions

### HLS Lightness Instead of Raw RGB

Lane markings are often brighter than asphalt. Using the lightness channel keeps the detector focused on intensity contrast while reducing sensitivity to color hue variation across cameras.

### Trapezoidal ROI

Forward-facing road cameras observe lanes in the lower half of the image. A trapezoidal mask encodes that prior directly and prevents the Hough transform from spending effort on sky, trees, signs, and vehicle edges.

### Weighted Line Averaging

Hough lines can be noisy and fragmented. RoadVision weights each candidate by segment length so stronger lane evidence has more influence than small edge fragments.

### Confidence Score

The confidence score is intentionally simple. It combines whether both lane sides were detected with how much candidate evidence supported the estimate. It is not a calibrated probability.

### Departure State

The departure heuristic compares the frame center against the lane center near the lower part of the image. This approximates the vehicle position under a center-mounted camera assumption.

## Error Handling

The backend handles common user and media problems:

- unsupported extension
- empty upload
- OpenCV failure to open file
- invalid video metadata
- missing output artifact

Unexpected failures are logged with stack traces through the structured logger and returned as API errors without console spam from the pipeline itself.

## Metrics

Each run records:

- input and output paths
- frame count and processed frame count
- resolution
- source FPS
- measured processing FPS
- average and max frame latency
- departure frame count
- low-confidence frame count

These numbers make the project benchmarkable and give recruiters a concrete signal that the system was tested like software, not only viewed as a visual demo.

## Known Limitations

This implementation assumes mostly straight lane boundaries in a forward-facing view. It does not yet perform camera calibration, inverse perspective mapping, temporal filtering, or semantic segmentation.

The failure modes are useful next-step material rather than hidden caveats:

- shadows and guardrails create false edges
- night footage weakens contrast
- curves do not fit a single straight line well
- faded paint can disappear from Canny edges
- off-center camera mounting changes the departure offset baseline

## Extension Plan

1. Add synthetic frame tests for centered, left drift, right drift, and no-lane scenes.
2. Add temporal smoothing to reduce frame-to-frame warning flicker.
3. Use perspective transform and polynomial fitting for curves.
4. Calibrate camera center offset per vehicle.
5. Add a segmentation model fallback for difficult lighting.
6. Add Docker and CI with smoke tests.
