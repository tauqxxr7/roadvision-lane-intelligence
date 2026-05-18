from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


@dataclass
class LaneDetectionResult:
    frame: np.ndarray
    departure_state: str
    lane_center_offset_px: Optional[float]
    confidence: float
    left_lane: Optional[Tuple[Point, Point]] = None
    right_lane: Optional[Tuple[Point, Point]] = None
    debug: dict = field(default_factory=dict)


@dataclass
class LaneDetectorConfig:
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 35
    min_line_length: int = 35
    max_line_gap: int = 120
    departure_offset_ratio: float = 0.12
    overlay_alpha: float = 0.35


class LaneDetector:
    """Classical lane detector optimized for explainability and fast CPU inference."""

    def __init__(self, config: LaneDetectorConfig | None = None) -> None:
        self.config = config or LaneDetectorConfig()

    def process_frame(self, frame: np.ndarray) -> LaneDetectionResult:
        if frame is None or frame.size == 0:
            raise ValueError("Received an empty frame for lane detection")

        height, width = frame.shape[:2]
        edges = self._edge_map(frame)
        roi = self._region_of_interest(edges)
        raw_lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )

        left_line, right_line, confidence = self._fit_lane_lines(raw_lines, width, height)
        overlay = self._draw_lane_overlay(frame, left_line, right_line)
        departure_state, offset_px = self._departure_state(left_line, right_line, width, height)

        return LaneDetectionResult(
            frame=overlay,
            departure_state=departure_state,
            lane_center_offset_px=offset_px,
            confidence=confidence,
            left_lane=left_line,
            right_lane=right_line,
            debug={
                "raw_line_count": 0 if raw_lines is None else len(raw_lines),
                "frame_width": width,
                "frame_height": height,
            },
        )

    def _edge_map(self, frame: np.ndarray) -> np.ndarray:
        # HLS lightness isolates painted markings better than raw RGB under mild illumination shifts.
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lightness = hls[:, :, 1]
        blurred = cv2.GaussianBlur(lightness, (5, 5), 0)
        return cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)

    def _region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        height, width = edges.shape[:2]
        polygon = np.array(
            [
                [
                    (int(0.08 * width), height),
                    (int(0.43 * width), int(0.58 * height)),
                    (int(0.57 * width), int(0.58 * height)),
                    (int(0.94 * width), height),
                ]
            ],
            dtype=np.int32,
        )
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _fit_lane_lines(
        self, raw_lines: Optional[np.ndarray], width: int, height: int
    ) -> Tuple[Optional[Tuple[Point, Point]], Optional[Tuple[Point, Point]], float]:
        if raw_lines is None:
            return None, None, 0.0

        left_candidates: list[Tuple[float, float, float]] = []
        right_candidates: list[Tuple[float, float, float]] = []

        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.45:
                continue

            intercept = y1 - slope * x1
            length = float(np.hypot(x2 - x1, y2 - y1))
            midpoint_x = (x1 + x2) / 2

            if slope < 0 and midpoint_x < width * 0.55:
                left_candidates.append((slope, intercept, length))
            elif slope > 0 and midpoint_x > width * 0.45:
                right_candidates.append((slope, intercept, length))

        y_bottom = height
        y_top = int(height * 0.60)
        left_line = self._weighted_line(left_candidates, y_bottom, y_top, width)
        right_line = self._weighted_line(right_candidates, y_bottom, y_top, width)

        detected = int(left_line is not None) + int(right_line is not None)
        raw_strength = min(len(left_candidates) + len(right_candidates), 12) / 12
        confidence = round((detected / 2) * 0.75 + raw_strength * 0.25, 3)
        return left_line, right_line, confidence

    def _weighted_line(
        self, candidates: list[Tuple[float, float, float]], y_bottom: int, y_top: int, width: int
    ) -> Optional[Tuple[Point, Point]]:
        if not candidates:
            return None

        weights = np.array([candidate[2] for candidate in candidates])
        slopes = np.array([candidate[0] for candidate in candidates])
        intercepts = np.array([candidate[1] for candidate in candidates])
        slope = float(np.average(slopes, weights=weights))
        intercept = float(np.average(intercepts, weights=weights))

        if abs(slope) < 1e-3:
            return None

        x_bottom = int((y_bottom - intercept) / slope)
        x_top = int((y_top - intercept) / slope)
        x_bottom = int(np.clip(x_bottom, 0, width - 1))
        x_top = int(np.clip(x_top, 0, width - 1))
        return (x_bottom, y_bottom), (x_top, y_top)

    def _draw_lane_overlay(
        self,
        frame: np.ndarray,
        left_line: Optional[Tuple[Point, Point]],
        right_line: Optional[Tuple[Point, Point]],
    ) -> np.ndarray:
        overlay = frame.copy()
        line_layer = np.zeros_like(frame)

        if left_line and right_line:
            polygon = np.array([[left_line[0], left_line[1], right_line[1], right_line[0]]], dtype=np.int32)
            cv2.fillPoly(line_layer, polygon, (40, 180, 90))

        for lane in (left_line, right_line):
            if lane:
                cv2.line(line_layer, lane[0], lane[1], (0, 245, 255), 8)

        overlay = cv2.addWeighted(overlay, 1.0, line_layer, self.config.overlay_alpha, 0)
        return overlay

    def _departure_state(
        self,
        left_line: Optional[Tuple[Point, Point]],
        right_line: Optional[Tuple[Point, Point]],
        width: int,
        height: int,
    ) -> Tuple[str, Optional[float]]:
        if not left_line or not right_line:
            return "LANE_NOT_CONFIDENT", None

        y_eval = int(height * 0.92)
        left_x = self._x_at_y(left_line, y_eval)
        right_x = self._x_at_y(right_line, y_eval)
        if left_x is None or right_x is None or right_x <= left_x:
            return "LANE_NOT_CONFIDENT", None

        lane_center = (left_x + right_x) / 2
        vehicle_center = width / 2
        offset_px = vehicle_center - lane_center
        threshold = width * self.config.departure_offset_ratio

        if offset_px > threshold:
            return "DRIFTING_LEFT", round(offset_px, 2)
        if offset_px < -threshold:
            return "DRIFTING_RIGHT", round(offset_px, 2)
        return "CENTERED", round(offset_px, 2)

    def _x_at_y(self, line: Tuple[Point, Point], y: int) -> Optional[float]:
        (x1, y1), (x2, y2) = line
        if y2 == y1:
            return None
        slope = (x2 - x1) / (y2 - y1)
        return x1 + slope * (y - y1)
