"""Detect prominent objects in a live camera feed with Gemini and OpenCV."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

if TYPE_CHECKING:
    from google import genai

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"
PROMPT = """Detect every prominent, distinct object in this image.
Return one entry per object, including a short label and its bounding box.
Use box_2d coordinates normalized to a 0-1000 scale in the exact order
[ymin, xmin, ymax, xmax]. Do not include objects that are only tiny or unclear.
"""
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "label": {"type": "STRING"},
            "box_2d": {
                "type": "ARRAY",
                "items": {"type": "NUMBER"},
            },
        },
        "required": ["label", "box_2d"],
    },
}


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


def detect_objects(frame: Any, client: genai.Client) -> list[dict[str, Any]]:
    """Send one camera frame to Gemini and return its detections."""
    from google.genai import types

    success, encoded_frame = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Could not encode camera frame")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=encoded_frame.tobytes(), mime_type="image/jpeg"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    detections = json.loads(response.text)
    if not isinstance(detections, list):
        raise ValueError("Gemini returned an unexpected detection format")
    return detections


def annotate_image(image: Any, detections: list[dict[str, Any]]) -> Any:
    """Draw detection boxes and labels on a camera frame."""
    height, width = image.shape[:2]
    for detection in detections:
        label = str(detection["label"])
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


def detect_local_objects(model: YOLO, frame: Any) -> list[dict[str, Any]]:
    """Detect objects locally with YOLO, without using the Gemini API."""
    result = model(frame, verbose=False)[0]
    height, width = frame.shape[:2]
    detections = []
    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < 0.4:
            continue
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        detections.append(
            {
                "label": f"{result.names[class_id]} {confidence:.0%}",
                "box_2d": [
                    y_min / height * 1000,
                    x_min / width * 1000,
                    y_max / height * 1000,
                    x_max / width * 1000,
                ],
            }
        )
    return detections


class DetectionWorker:
    """Run Gemini requests in the background without blocking the camera preview."""

    def __init__(self, client: genai.Client, interval: float) -> None:
        self.client = client
        self.interval = max(interval, 1.0)
        self._frame: Any = None
        self._detections: list[dict[str, Any]] = []
        self._error = ""
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._has_frame = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame: Any) -> None:
        with self._lock:
            self._frame = frame.copy()
        self._has_frame.set()

    def result(self) -> tuple[list[dict[str, Any]], str]:
        with self._lock:
            return self._detections.copy(), self._error

    def stop(self) -> None:
        self._stop_event.set()
        self._has_frame.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        next_request = 0.0
        while not self._stop_event.is_set():
            self._has_frame.wait(timeout=0.2)
            if self._stop_event.is_set():
                break

            wait_time = next_request - time.monotonic()
            if wait_time > 0:
                self._stop_event.wait(wait_time)
                continue

            with self._lock:
                frame = self._frame
            if frame is None:
                continue

            try:
                detections = detect_objects(frame, self.client)
                with self._lock:
                    self._detections = detections
                    self._error = ""
            except Exception as error:
                with self._lock:
                    self._error = f"Gemini: {error}"
            finally:
                next_request = time.monotonic() + self.interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--interval", type=float, default=60.0,
        help="Seconds between Gemini requests (default: 60.0)",
    )
    parser.add_argument(
        "--no-gemini", action="store_true",
        help="Use only local YOLO detection and make no Gemini requests",
    )
    parser.add_argument(
        "--yolo-model", type=Path, default=Path("yolo11n.pt"),
        help="Path to the local YOLO weights file (default: yolo11n.pt)",
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
    worker = None
    if not args.no_gemini:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            from google import genai

            worker = DetectionWorker(genai.Client(api_key=api_key), args.interval)
            worker.start()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise SystemExit("Could not read a frame from the camera")

            local_detections = detect_local_objects(local_model, frame)
            annotated = annotate_image(frame, local_detections)
            error = ""
            if worker:
                worker.submit(frame)
                gemini_detections, error = worker.result()
                annotated = annotate_image(annotated, gemini_detections)
            elif not args.no_gemini:
                error = "Gemini disabled: GEMINI_API_KEY is not set"
            if error:
                cv2.putText(
                    annotated,
                    error[:100],
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Gemini object detections - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if worker:
            worker.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
