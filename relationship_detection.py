"""Deterministic spatial relationship detection for tracked objects."""

from __future__ import annotations

from datetime import datetime, timezone
from math import hypot
from typing import Any, Protocol


class EventSink(Protocol):
    """Minimal event-log interface required by the relationship detector."""

    def append(self, event: dict[str, Any]) -> None:
        ...


class RelationshipDetector:
    """Compare active tracked objects and emit relationship state changes."""

    def __init__(
        self,
        event_log: EventSink,
        near_threshold: float = 100.0,
        far_threshold: float = 200.0,
        directional_margin: float = 10.0,
    ) -> None:
        if near_threshold < 0 or far_threshold < near_threshold:
            raise ValueError("Require 0 <= near_threshold <= far_threshold")
        if directional_margin < 0:
            raise ValueError("directional_margin must be non-negative")
        self.event_log = event_log
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.directional_margin = directional_margin
        self.previous_relationships: dict[tuple[Any, Any], set[str]] = {}

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compare one frame of active detections with the previous frame."""
        objects = {
            detection["object_id"]: detection
            for detection in detections
            if detection.get("object_id") is not None
        }
        current_relationships: dict[tuple[Any, Any], set[str]] = {}
        events: list[dict[str, Any]] = []

        object_ids = sorted(objects, key=str)
        for subject_id in object_ids:
            for object_id in object_ids:
                if subject_id == object_id:
                    continue
                key = (subject_id, object_id)
                previous = self.previous_relationships.get(key, set())
                relations = self._relationships_for_pair(
                    objects[subject_id], objects[object_id], previous
                )
                current_relationships[key] = relations
                events.extend(
                    self._state_events(
                        objects[subject_id],
                        objects[object_id],
                        relations,
                        previous,
                    )
                )

        self.previous_relationships = current_relationships
        for event in events:
            self.event_log.append(event)
        return events

    def active_relationships(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the current relationship state as serializable context."""
        objects = {
            detection["object_id"]: detection
            for detection in detections
            if detection.get("object_id") is not None
        }
        relationships = []
        for (subject_id, object_id), relations in self.previous_relationships.items():
            if subject_id not in objects or object_id not in objects:
                continue
            for relation in sorted(relations):
                relationships.append({
                    "subject_id": subject_id,
                    "subject_class": objects[subject_id]["label_class"],
                    "relation": relation,
                    "object_id": object_id,
                    "object_class": objects[object_id]["label_class"],
                })
        return relationships

    def _relationships_for_pair(
        self,
        subject: dict[str, Any],
        object_: dict[str, Any],
        previous: set[str],
    ) -> set[str]:
        subject_x, subject_y = subject["center"]
        object_x, object_y = object_["center"]
        delta_x = object_x - subject_x
        delta_y = object_y - subject_y
        relations: set[str] = set()

        axis_margin = self.directional_margin
        if abs(delta_x) > abs(delta_y) + axis_margin:
            relations.add("RIGHT_OF" if delta_x > 0 else "LEFT_OF")
        elif abs(delta_y) > abs(delta_x) + axis_margin:
            relations.add("BELOW" if delta_y > 0 else "ABOVE")
        else:
            relations.update(previous & {"LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW"})

        distance = hypot(delta_x, delta_y)
        if distance <= self.near_threshold:
            relations.add("NEAR")
        elif distance >= self.far_threshold:
            relations.add("FAR")
        else:
            relations.update(previous & {"NEAR", "FAR"})
        return relations

    def _state_events(
        self,
        subject: dict[str, Any],
        object_: dict[str, Any],
        current: set[str],
        previous: set[str],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for relation in sorted(current - previous):
            event_type = "RELATIONSHIP_CREATED"
            if previous & self._same_axis(relation):
                event_type = "RELATIONSHIP_CHANGED"
            events.append(self._event(event_type, subject, relation, object_))
        for relation in sorted(previous - current):
            events.append(self._event("RELATIONSHIP_REMOVED", subject, relation, object_))
        return events

    @staticmethod
    def _same_axis(relation: str) -> set[str]:
        if relation in {"LEFT_OF", "RIGHT_OF"}:
            return {"LEFT_OF", "RIGHT_OF"}
        if relation in {"ABOVE", "BELOW"}:
            return {"ABOVE", "BELOW"}
        if relation in {"NEAR", "FAR"}:
            return {"NEAR", "FAR"}
        return {relation}

    def _event(
        self,
        event_type: str,
        subject: dict[str, Any],
        relation: str,
        object_: dict[str, Any],
    ) -> dict[str, Any]:
        subject_center = subject["center"]
        object_center = object_["center"]
        distance = hypot(
            subject_center[0] - object_center[0],
            subject_center[1] - object_center[1],
        )
        if relation in {"LEFT_OF", "RIGHT_OF"}:
            margin = abs(subject_center[0] - object_center[0]) - abs(
                subject_center[1] - object_center[1]
            )
            confidence = margin / (abs(subject_center[0] - object_center[0]) + 1.0)
        elif relation in {"ABOVE", "BELOW"}:
            margin = abs(subject_center[1] - object_center[1]) - abs(
                subject_center[0] - object_center[0]
            )
            confidence = margin / (abs(subject_center[1] - object_center[1]) + 1.0)
        elif relation == "NEAR":
            confidence = (self.near_threshold - distance) / (self.near_threshold + 1.0)
        else:
            confidence = (distance - self.far_threshold) / (distance + 1.0)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "subject_id": subject["object_id"],
            "subject_class": subject["label_class"],
            "relation": relation,
            "object_id": object_["object_id"],
            "object_class": object_["label_class"],
            "confidence": max(0.0, min(1.0, confidence)),
            "subject_position": subject_center,
            "object_position": object_center,
        }
        previous_relations = self._previous_relation(event_type, relation)
        if previous_relations:
            event["previous_relation"] = next(iter(previous_relations))
        return event

    @staticmethod
    def _previous_relation(event_type: str, relation: str) -> set[str]:
        if event_type != "RELATIONSHIP_CHANGED":
            return set()
        if relation in {"LEFT_OF", "RIGHT_OF"}:
            return {"RIGHT_OF" if relation == "LEFT_OF" else "LEFT_OF"}
        if relation in {"ABOVE", "BELOW"}:
            return {"BELOW" if relation == "ABOVE" else "ABOVE"}
        if relation in {"NEAR", "FAR"}:
            return {"FAR" if relation == "NEAR" else "NEAR"}
        return set()
