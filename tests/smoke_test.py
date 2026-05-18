from __future__ import annotations

import cv2
import numpy as np
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.lane_detector import LaneDetector


def main() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(frame, (180, 480), (290, 280), (255, 255, 255), 8)
    cv2.line(frame, (460, 480), (350, 280), (255, 255, 255), 8)

    result = LaneDetector().process_frame(frame)
    if result.departure_state != "CENTERED":
        raise AssertionError(f"Expected CENTERED, got {result.departure_state}")
    if result.confidence < 0.5:
        raise AssertionError(f"Expected confidence >= 0.5, got {result.confidence}")

    print(f"smoke ok: state={result.departure_state}, confidence={result.confidence}")


if __name__ == "__main__":
    main()
