"""Detect prominent objects in a live camera feed with Gemini and OpenCV."""

from __future__ import annotations

import argparse
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from event_storage import JsonlEventLogger
from gemini_event_trigger import EventTrigger, GeminiEventWorker
from relationship_detection import RelationshipDetector
from temporal_reasoning import TemporalReasoner

load_dotenv()

class DetectionLog:
    """Keep structured detection events in memory for later processing."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._next_temporary_id = 1
        self._lock = threading.Lock()

    def append(
        self,
        object_id: Any,
        object_class: str,
        confidence: float,
        bounding_box: list[float],
        frame_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Append one JSON-serializable detection event and return it."""
        with self._lock:
            if object_id is None:
                object_id = f"temp-{self._next_temporary_id:06d}"
                self._next_temporary_id += 1

            x1, y1, x2, y2 = bounding_box
            width, height = frame_size
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            record = {
                "event_type": "detection",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "object_id": object_id,
                "object": {
                    "id": object_id,
                    "class": object_class,
                },
                "class": object_class,
                "confidence": confidence,
                "bounding_box": bounding_box,
                "center": center,
                "center_normalized": [center[0] / width, center[1] / height],
            }
            self.records.append(record)
            return record


DETECTION_LOG = DetectionLog()


class DetectionEventLog:
    """Keep structured object-state events in memory for later processing."""

    def __init__(self, file_logger: JsonlEventLogger | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self.file_logger = file_logger
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        """Append one JSON-serializable event to the in-memory stream."""
        with self._lock:
            self.records.append(event)
            if self.file_logger is not None:
                self.file_logger.append(event)


EVENT_FILE_LOGGER = JsonlEventLogger()
EVENT_LOG = DetectionEventLog(EVENT_FILE_LOGGER)


def pixel_box(box_2d: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert Gemini's [ymin, xmin, ymax, xmax] box to pixel coordinates."""
    if len(box_2d) != 4:
        raise ValueError(f"Expected four coordinates, got {box_2d!r}")

    ymin, xmin, ymax, xmax = box_2d
    left = round(max(0, min(1000, xmin)) * width / 1000)
    top = round(max(0, min(1000, ymin)) * height / 1000)
    right = round(max(0, min(1000, xmax)) * width / 1000)
    bottom = round(max(0, min(1000, ymax)) * height / 1000)
    return left, top, right, bottom


def annotate_image(image: Any, detections: list[dict[str, Any]]) -> Any:
    """Draw detection boxes and labels on a camera frame."""
    height, width = image.shape[:2]
    for detection in detections:
        object_id = detection.get("object_id")
        label = str(detection["label"])
        if object_id is not None:
            label = f"{label} (ID: {object_id})"
        left, top, right, bottom = pixel_box(detection["box_2d"], width, height)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return image


class YOLOTracker:
    """Run persistent ByteTrack tracking and retain per-object center history."""

    def __init__(self, model: YOLO) -> None:
        self.model = model
        self.history: dict[int | str, list[list[float]]] = {}

    def update(self, frame: Any) -> Any:
        """Track objects in one frame; ByteTrack handles new and missing objects."""
        return self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

    def record_center(self, object_id: int | str, center: list[float]) -> None:
        """Store a pixel-center sample for later movement and temporal reasoning."""
        self.history.setdefault(object_id, []).append(center)


class FrameEventDetector:
    """Compare consecutive tracked frames and emit object-state change events."""

    def __init__(
        self,
        event_log: DetectionEventLog = EVENT_LOG,
        movement_threshold: float = 20.0,
    ) -> None:
        if movement_threshold < 0:
            raise ValueError("movement_threshold must be non-negative")
        self.event_log = event_log
        self.movement_threshold = movement_threshold
        self.previous_state: dict[Any, dict[str, Any]] = {}

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Emit events for the current frame and return the newly emitted events."""
        current_state = {
            detection["object_id"]: detection
            for detection in detections
            if detection.get("object_id") is not None
        }
        events: list[dict[str, Any]] = []

        for object_id, current in current_state.items():
            previous = self.previous_state.get(object_id)
            if previous is None:
                events.append(self._event("OBJECT_APPEARED", object_id, current))
                continue

            if current["label_class"] != previous["label_class"]:
                events.append(
                    self._event(
                        "OBJECT_CLASS_CHANGED",
                        object_id,
                        current,
                        previous,
                    )
                )
            if self._distance(current["center"], previous["center"]) > self.movement_threshold:
                events.append(
                    self._event("OBJECT_MOVED", object_id, current, previous)
                )

        for object_id, previous in self.previous_state.items():
            if object_id not in current_state:
                events.append(
                    self._event("OBJECT_DISAPPEARED", object_id, previous)
                )

        self.previous_state = current_state
        for event in events:
            self.event_log.append(event)
        return events

    @staticmethod
    def _distance(first: list[float], second: list[float]) -> float:
        return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5

    @staticmethod
    def _event(
        event_type: str,
        object_id: Any,
        current: dict[str, Any],
        previous: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "object_id": object_id,
            "object_class": current["label_class"],
            "confidence": current["confidence"],
            "previous_position": None if previous is None else previous["center"],
            "previous_bounding_box": (
                None if previous is None else previous["bounding_box"]
            ),
            "current_position": current["center"],
            "current_bounding_box": current["bounding_box"],
        }
        if event_type == "OBJECT_DISAPPEARED":
            event["confidence"] = previous["confidence"] if previous else None
            event["current_position"] = None
            event["current_bounding_box"] = None
        if event_type == "OBJECT_CLASS_CHANGED" and previous is not None:
            event["previous_class"] = previous["label_class"]
        return event


def detect_local_objects(
    tracker: YOLOTracker,
    frame: Any,
    detection_log: DetectionLog = DETECTION_LOG,
) -> list[dict[str, Any]]:
    """Track objects locally with YOLO and return the existing annotation shape."""
    result = tracker.update(frame)
    height, width = frame.shape[:2]
    detections = []
    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < 0.4:
            continue
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        object_class = str(result.names[class_id])
        track_ids = box.id
        object_id = None if track_ids is None else int(track_ids[0].item())
        bounding_box = [x_min, y_min, x_max, y_max]
        center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
        record = detection_log.append(
            object_id,
            object_class,
            confidence,
            bounding_box,
            (width, height),
        )
        object_id = record["object_id"]
        tracker.record_center(object_id, center)
        detections.append(
            {
                "label": f"{object_class} {confidence:.0%}",
                "object_id": object_id,
                "label_class": object_class,
                "confidence": confidence,
                "bounding_box": bounding_box,
                "center": center,
                "box_2d": [
                    y_min / height * 1000,
                    x_min / width * 1000,
                    y_max / height * 1000,
                    x_max / width * 1000,
                ],
            }
        )
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--no-gemini", action="store_true",
        help="Use only local YOLO detection and make no Gemini requests",
    )
    parser.add_argument(
        "--yolo-model", type=Path, default=Path("yolo11n.pt"),
        help="Path to the local YOLO weights file (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--movement-threshold", type=float, default=20.0,
        help="Pixel distance required to emit OBJECT_MOVED (default: 20.0)",
    )
    parser.add_argument(
        "--near-threshold", type=float, default=100.0,
        help="Pixel distance at or below which objects are NEAR (default: 100.0)",
    )
    parser.add_argument(
        "--far-threshold", type=float, default=200.0,
        help="Pixel distance at or above which objects are FAR (default: 200.0)",
    )
    parser.add_argument(
        "--directional-margin", type=float, default=10.0,
        help="Pixel margin for stable directional relationships (default: 10.0)",
    )
    parser.add_argument(
        "--temporal-window", type=float, default=5.0,
        help="Seconds used to correlate temporal events (default: 5.0)",
    )
    parser.add_argument(
        "--stationary-frames", type=int, default=2,
        help="Stationary frames required before PICKED_UP (default: 2)",
    )
    parser.add_argument(
        "--gemini-cooldown", type=float, default=10.0,
        help="Minimum seconds between Gemini calls (default: 10.0)",
    )
    parser.add_argument(
        "--gemini-min-confidence", type=float, default=0.4,
        help="Minimum event confidence for Gemini triggers (default: 0.4)",
    )
    parser.add_argument(
        "--gemini-history-window", type=float, default=30.0,
        help="Seconds of event history sent to Gemini (default: 30.0)",
    )
    parser.add_argument(
        "--gemini-semantic-min-confidence", type=float, default=0.6,
        help="Minimum confidence for accepted Gemini events (default: 0.6)",
    )
    args = parser.parse_args()

    try:
        local_model = YOLO(str(args.yolo_model))
    except Exception as error:
        raise SystemExit(
            f"Could not load YOLO weights from {args.yolo_model}. "
            "Download yolo11n.pt from "
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt "
            "and place it in this project folder, or pass --yolo-model PATH."
        ) from error
    tracker = YOLOTracker(local_model)
    event_detector = FrameEventDetector(
        movement_threshold=args.movement_threshold,
    )
    relationship_detector = RelationshipDetector(
        EVENT_LOG,
        near_threshold=args.near_threshold,
        far_threshold=args.far_threshold,
        directional_margin=args.directional_margin,
    )
    temporal_reasoner = TemporalReasoner(
        EVENT_LOG,
        time_window=args.temporal_window,
        movement_threshold=args.movement_threshold,
        carry_distance=args.near_threshold,
        stationary_frames=args.stationary_frames,
    )
    worker = None
    if not args.no_gemini:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            from google import genai

            worker = GeminiEventWorker(
                genai.Client(api_key=api_key),
                EVENT_LOG,
                EventTrigger(
                    cooldown=args.gemini_cooldown,
                    minimum_confidence=args.gemini_min_confidence,
                ),
                history_window=args.gemini_history_window,
                semantic_minimum_confidence=args.gemini_semantic_min_confidence,
            )
            worker.start()
        elif not args.no_gemini:
            print("Gemini disabled: GEMINI_API_KEY is not set", file=sys.stderr)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise SystemExit("Could not read a frame from the camera")

            local_detections = detect_local_objects(tracker, frame)
            raw_events = event_detector.update(local_detections)
            raw_events.extend(relationship_detector.update(local_detections))
            temporal_events = temporal_reasoner.update(local_detections, raw_events)
            raw_events.extend(temporal_events)
            annotated = annotate_image(frame, local_detections)
            if worker:
                worker.submit(
                    frame,
                    raw_events,
                    local_detections,
                    relationship_detector.active_relationships(local_detections),
                    list(EVENT_LOG.records),
                )
                worker.result()
            cv2.imshow("Gemini object detections - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if worker:
            worker.stop()
        EVENT_FILE_LOGGER.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
