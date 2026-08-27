"""Event-triggered Gemini semantic interpretation."""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:
    from google import genai


SEMANTIC_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "event_type": {"type": "STRING"},
            "description": {"type": "STRING"},
            "subject_id": {"type": "STRING"},
            "subject_class": {"type": "STRING"},
            "object_id": {"type": "STRING"},
            "object_class": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
            "evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["event_type", "description", "confidence", "evidence"],
    },
}


class EventTrigger:
    """Apply configurable event significance, confidence, and cooldown rules."""

    DEFAULT_EVENT_TYPES = {
        "OBJECT_APPEARED",
        "OBJECT_DISAPPEARED",
        "OBJECT_MOVED",
        "OBJECT_CLASS_CHANGED",
        "RELATIONSHIP_CHANGED",
        "RELATIONSHIP_CREATED",
        "RELATIONSHIP_REMOVED",
        "PICKED_UP",
        "PLACED_DOWN",
        "MOVED_FROM_TO",
        "ENTERED",
        "LEFT",
        "APPROACHED",
        "MOVED_AWAY",
    }

    def __init__(
        self,
        cooldown: float = 10.0,
        minimum_confidence: float = 0.0,
        event_types: set[str] | None = None,
    ) -> None:
        if cooldown < 0 or not 0 <= minimum_confidence <= 1:
            raise ValueError("Invalid Gemini trigger cooldown or confidence")
        self.cooldown = cooldown
        self.minimum_confidence = minimum_confidence
        self.event_types = event_types or self.DEFAULT_EVENT_TYPES.copy()
        self._last_trigger = 0.0

    def should_trigger(
        self, events: list[dict[str, Any]], detections: list[dict[str, Any]]
    ) -> bool:
        """Return true only when a significant event is present and cooldown allows it."""
        if time.monotonic() - self._last_trigger < self.cooldown:
            return False
        significant = any(
            event.get("event_type") in self.event_types
            and float(event.get("confidence", 1.0) or 0.0) >= self.minimum_confidence
            for event in events
        )
        ambiguous = any(
            float(detection.get("confidence", 1.0)) < self.minimum_confidence
            for detection in detections
        )
        if significant or ambiguous:
            self._last_trigger = time.monotonic()
            return True
        return False


class GeminiEventWorker:
    """Send selected frames and bounded event context to Gemini in the background."""

    ALLOWED_SEMANTIC_EVENTS = {
        "PERSON_PICKED_UP_OBJECT",
        "PERSON_PUT_DOWN_OBJECT",
        "PERSON_OPENED_DOOR",
        "PERSON_CLOSED_DOOR",
        "PERSON_ENTERED_AREA",
        "PERSON_LEFT_AREA",
        "PERSON_INTERACTED_WITH_OBJECT",
        "OBJECT_BEING_USED",
    }

    def __init__(
        self,
        client: genai.Client,
        event_log: Any,
        trigger: EventTrigger,
        history_window: float = 30.0,
        semantic_minimum_confidence: float = 0.6,
    ) -> None:
        if history_window <= 0 or not 0 <= semantic_minimum_confidence <= 1:
            raise ValueError("Invalid Gemini history window or semantic confidence")
        self.client = client
        self.event_log = event_log
        self.trigger = trigger
        self.history_window = history_window
        self.semantic_minimum_confidence = semantic_minimum_confidence
        self._recent_semantic: dict[tuple[Any, ...], float] = {}
        self._pending: tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None = None
        self._results: list[dict[str, Any]] = []
        self._error = ""
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._has_work = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(
        self,
        frame: Any,
        events: list[dict[str, Any]],
        objects: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        event_history: list[dict[str, Any]],
    ) -> None:
        """Queue a frame only when trigger policy identifies significant context."""
        if not self.trigger.should_trigger(events, objects):
            return
        cutoff = time.time() - self.history_window
        recent_history = [
            event for event in event_history
            if self._event_epoch(event) >= cutoff
        ]
        with self._lock:
            self._pending = (frame.copy(), recent_history, objects.copy(), relationships.copy())
        self._has_work.set()

    def result(self) -> tuple[list[dict[str, Any]], str]:
        with self._lock:
            results = self._results.copy()
            self._results.clear()
            return results, self._error

    def stop(self) -> None:
        self._stop_event.set()
        self._has_work.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._has_work.wait(timeout=0.2)
            if self._stop_event.is_set():
                return
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                self._has_work.clear()
                continue
            self._has_work.clear()
            try:
                frame, history, objects, relationships = pending
                events = self._interpret(frame, history, objects, relationships)
                for event in events:
                    self.event_log.append(event)
                with self._lock:
                    self._results.extend(events)
                    self._error = ""
            except Exception as error:
                with self._lock:
                    self._error = f"Gemini: {error}"
                print(self._error, file=sys.stderr)

    def _interpret(
        self,
        frame: Any,
        history: list[dict[str, Any]],
        objects: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        from google.genai import types

        success, encoded_frame = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Could not encode camera frame")
        prompt = (
            "Interpret only the significant scene change represented by this frame and context. "
            "Reason over both the image and supplied evidence. Return ONLY a JSON array of "
            "higher-level semantic events that geometry alone cannot reliably determine. "
            "Do not invent events or return natural language. Allowed event_type values are: "
            + ", ".join(sorted(self.ALLOWED_SEMANTIC_EVENTS))
            + ". Use exact tracked IDs and confidence from 0 to 1. Each event must include "
            "event_type, description, confidence, evidence, and subject/object IDs and classes "
            "when applicable. Context:\n"
            + json.dumps({"recent_events": history, "objects": objects, "relationships": relationships})
        )
        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                types.Part.from_bytes(data=encoded_frame.tobytes(), mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SEMANTIC_RESPONSE_SCHEMA,
            ),
        )
        parsed = json.loads(response.text)
        if not isinstance(parsed, list):
            raise ValueError("Gemini returned an unexpected semantic event format")
        return self._validate_events(parsed, objects)

    def _validate_events(
        self, parsed: list[Any], objects: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        valid_ids = {str(item.get("object_id")) for item in objects}
        now = time.monotonic()
        self._recent_semantic = {
            key: timestamp
            for key, timestamp in self._recent_semantic.items()
            if now - timestamp <= self.history_window
        }
        normalized: list[dict[str, Any]] = []
        for event in parsed:
            if not isinstance(event, dict):
                continue
            event_type = event.get("event_type")
            description = event.get("description")
            confidence = event.get("confidence")
            evidence = event.get("evidence")
            if (
                event_type not in self.ALLOWED_SEMANTIC_EVENTS
                or not isinstance(description, str)
                or not description.strip()
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
                or confidence < self.semantic_minimum_confidence
                or not isinstance(evidence, list)
                or not evidence
                or not all(isinstance(item, str) and item.strip() for item in evidence)
            ):
                continue
            subject_id = event.get("subject_id")
            object_id = event.get("object_id")
            if subject_id is not None and str(subject_id) not in valid_ids:
                continue
            if object_id is not None and str(object_id) not in valid_ids:
                continue
            subject_class = event.get("subject_class")
            object_class = event.get("object_class")
            if subject_id is not None and not isinstance(subject_class, str):
                continue
            if object_id is not None and not isinstance(object_class, str):
                continue
            key = (event_type, subject_id, object_id)
            if key in self._recent_semantic:
                continue
            self._recent_semantic[key] = now
            normalized.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "subject_id": subject_id,
                "subject_class": subject_class,
                "object_id": object_id,
                "object_class": object_class,
                "description": description.strip(),
                "confidence": float(confidence),
                "source": "gemini",
                "evidence": evidence,
            })
        return normalized

    @staticmethod
    def _event_epoch(event: dict[str, Any]) -> float:
        return datetime.fromisoformat(event["timestamp"]).timestamp()
