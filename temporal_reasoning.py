"""Conservative, rule-based temporal reasoning over tracked event history."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from math import hypot
from typing import Any, Protocol


class EventSink(Protocol):
    def append(self, event: dict[str, Any]) -> None:
        ...


class TemporalReasoner:
    """Infer higher-level actions from recent raw events and object states."""

    def __init__(
        self,
        event_log: EventSink,
        time_window: float = 5.0,
        movement_threshold: float = 20.0,
        carry_distance: float = 100.0,
        stationary_frames: int = 2,
    ) -> None:
        if time_window <= 0 or movement_threshold < 0 or carry_distance < 0:
            raise ValueError("Temporal windows and spatial thresholds are invalid")
        if stationary_frames < 1:
            raise ValueError("stationary_frames must be at least 1")
        self.event_log = event_log
        self.time_window = time_window
        self.movement_threshold = movement_threshold
        self.carry_distance = carry_distance
        self.stationary_frames = stationary_frames
        self.event_history: deque[dict[str, Any]] = deque()
        self.state_history: deque[dict[str, Any]] = deque()
        self._previous_states: dict[Any, dict[str, Any]] = {}
        self._stationary_counts: dict[Any, int] = {}
        self._pickup_pairs: dict[Any, Any] = {}
        self._emitted: set[tuple[Any, ...]] = set()

    def update(
        self, detections: list[dict[str, Any]], raw_events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Record one frame and infer only supported multi-event sequences."""
        now = datetime.now(timezone.utc)
        self._expire(now)
        self.event_history.extend(raw_events)
        current_states = {
            detection["object_id"]: detection
            for detection in detections
            if detection.get("object_id") is not None
        }
        self.state_history.append({
            "timestamp": now.isoformat(), "objects": current_states.copy()
        })
        inferred = self._infer_event_sequences(raw_events)
        inferred.extend(self._infer_carry(current_states))
        inferred.extend(self._infer_placed_down(current_states))
        self._previous_states = current_states
        self.event_history.extend(inferred)
        for event in inferred:
            self.event_log.append(event)
        return inferred

    def _infer_event_sequences(self, raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        inferred: list[dict[str, Any]] = []
        for event in raw_events:
            event_type = event.get("event_type")
            object_id = event.get("object_id")
            if event_type == "OBJECT_MOVED":
                appearance = self._find_recent("OBJECT_APPEARED", object_id)
                if appearance and self._distance(
                    appearance["current_position"], event["current_position"]
                ) >= self.movement_threshold:
                    key = ("ENTERED", object_id, appearance["timestamp"])
                    if key not in self._emitted:
                        self._emitted.add(key)
                        inferred.append(self._action_event(
                            "ENTERED", event, appearance, [appearance, event]
                        ))
                moves = [item for item in self.event_history
                         if item.get("event_type") == "OBJECT_MOVED"
                         and item.get("object_id") == object_id]
                if len(moves) >= 2:
                    first = moves[-2]
                    key = ("MOVED_FROM_TO", object_id, first["timestamp"], event["timestamp"])
                    if key not in self._emitted:
                        self._emitted.add(key)
                        inferred.append(self._action_event(
                            "MOVED_FROM_TO", event, first, [first, event]
                        ))
            elif event_type == "OBJECT_DISAPPEARED":
                key = ("LEFT", object_id, event["timestamp"])
                if key not in self._emitted:
                    self._emitted.add(key)
                    inferred.append(self._action_event("LEFT", event, None, [event]))
            elif event_type == "RELATIONSHIP_CHANGED":
                pair = (event.get("previous_relation"), event.get("relation"))
                action = {
                    ("FAR", "NEAR"): "APPROACHED",
                    ("NEAR", "FAR"): "MOVED_AWAY",
                }.get(pair)
                if action:
                    key = (action, event["subject_id"], event["object_id"], event["timestamp"])
                    if key not in self._emitted:
                        self._emitted.add(key)
                        inferred.append(self._relationship_action_event(action, event))
        return inferred

    def _infer_carry(self, current_states: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
        inferred: list[dict[str, Any]] = []
        stationary_before = {
            object_id: count >= self.stationary_frames
            for object_id, count in self._stationary_counts.items()
        }
        for object_id, current in current_states.items():
            previous = self._previous_states.get(object_id)
            displacement = 0.0 if previous is None else self._distance(
                current["center"], previous["center"]
            )
            if displacement <= self.movement_threshold:
                self._stationary_counts[object_id] = self._stationary_counts.get(object_id, 0) + 1
                continue
            self._stationary_counts[object_id] = 0
            if not stationary_before.get(object_id, False):
                continue
            if current["label_class"].lower() == "person":
                continue
            previous_object = self._previous_states.get(object_id)
            if previous_object is None:
                continue
            for person_id, person in current_states.items():
                if person_id == object_id or person["label_class"].lower() != "person":
                    continue
                previous_person = self._previous_states.get(person_id)
                if previous_person is None:
                    continue
                person_displacement = self._distance(person["center"], previous_person["center"])
                distance = self._distance(current["center"], person["center"])
                alignment = self._vector_alignment(previous_object, current, previous_person, person)
                if distance <= self.carry_distance and person_displacement > self.movement_threshold and alignment >= 0.7:
                    key = ("PICKED_UP", object_id, person_id)
                    if key not in self._emitted:
                        self._emitted.add(key)
                        self._pickup_pairs[object_id] = person_id
                        inferred.append({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "event_type": "PICKED_UP",
                            "subject_id": object_id,
                            "subject_class": current["label_class"],
                            "object_id": person_id,
                            "object_class": person["label_class"],
                            "previous_position": previous_object["center"],
                            "current_position": current["center"],
                            "confidence": max(0.0, min(1.0, 0.5 * alignment + 0.5 * (1 - distance / (self.carry_distance + 1)))),
                            "source": "temporal_reasoning",
                            "evidence": ["OBJECT_MOVED", "NEAR"],
                        })
        return inferred

    def _infer_placed_down(self, current_states: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
        inferred: list[dict[str, Any]] = []
        for object_id, person_id in list(self._pickup_pairs.items()):
            current = current_states.get(object_id)
            person = current_states.get(person_id)
            if current is None or self._stationary_counts.get(object_id, 0) < self.stationary_frames:
                continue
            key = ("PLACED_DOWN", object_id, person_id)
            if key in self._emitted:
                continue
            self._emitted.add(key)
            inferred.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "PLACED_DOWN",
                "subject_id": object_id,
                "subject_class": current["label_class"],
                "object_id": person_id,
                "object_class": None if person is None else person["label_class"],
                "previous_position": None,
                "current_position": current["center"],
                "confidence": 0.7,
                "source": "temporal_reasoning",
                "evidence": ["PICKED_UP", "OBJECT_MOVED"],
            })
        return inferred

    def _find_recent(self, event_type: str, object_id: Any) -> dict[str, Any] | None:
        for event in reversed(self.event_history):
            if event.get("event_type") == event_type and event.get("object_id") == object_id:
                return event
        return None

    def _expire(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.time_window)
        while self.event_history and datetime.fromisoformat(self.event_history[0]["timestamp"]) < cutoff:
            self.event_history.popleft()
        while self.state_history and datetime.fromisoformat(self.state_history[0]["timestamp"]) < cutoff:
            self.state_history.popleft()

    @staticmethod
    def _distance(first: list[float], second: list[float]) -> float:
        return hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _vector_alignment(previous: dict[str, Any], current: dict[str, Any], previous_person: dict[str, Any], person: dict[str, Any]) -> float:
        first = [current["center"][i] - previous["center"][i] for i in range(2)]
        second = [person["center"][i] - previous_person["center"][i] for i in range(2)]
        first_length, second_length = hypot(*first), hypot(*second)
        if first_length == 0 or second_length == 0:
            return 0.0
        return max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)))

    @staticmethod
    def _action_event(event_type: str, event: dict[str, Any], previous: dict[str, Any] | None, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "subject_id": event["object_id"],
            "subject_class": event["object_class"],
            "previous_position": event.get("previous_position") if previous is None else previous.get("current_position"),
            "current_position": event.get("current_position"),
            "confidence": event.get("confidence"),
            "source": "temporal_reasoning",
            "evidence": [item["event_type"] for item in evidence],
        }

    @staticmethod
    def _relationship_action_event(event_type: str, event: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "subject_id": event["subject_id"],
            "subject_class": event["subject_class"],
            "object_id": event["object_id"],
            "object_class": event["object_class"],
            "previous_position": None,
            "current_position": event.get("subject_position"),
            "object_position": event.get("object_position"),
            "confidence": event["confidence"],
            "source": "temporal_reasoning",
            "evidence": ["RELATIONSHIP_CHANGED"],
        }
