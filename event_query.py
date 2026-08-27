"""Retrieval and answering layers for the persisted vision event stream."""

from __future__ import annotations

import json
import os
import re
from difflib import get_close_matches
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol


class LanguageModel(Protocol):
    def answer(self, question: str, evidence: dict[str, Any]) -> str:
        ...


@dataclass
class RetrievalResult:
    events: list[dict[str, Any]]
    objects: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    time_description: str = "available history"

    def as_context(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "current_objects": self.objects,
            "current_relationships": self.relationships,
            "time_scope": self.time_description,
        }


class JsonlEventStore:
    """Read-only event source that can later be replaced by a database adapter."""

    def __init__(self, path: str | Path = "events.jsonl") -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        """Load valid JSON objects, skipping incomplete lines safely."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as event_file:
            for line in event_file:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and isinstance(event.get("timestamp"), str):
                    records.append(event)
        return records


class EventRetriever:
    """Select only evidence relevant to a natural-language question."""

    ENTITY_ALIASES = {
        "phone": {"phone", "cell phone", "mobile", "cellphone"},
        "cell phone": {"phone", "cell phone", "mobile", "cellphone"},
        "mobile": {"phone", "cell phone", "mobile", "cellphone"},
        "person": {"person", "people", "human", "humans"},
    }

    def __init__(self, store: JsonlEventStore) -> None:
        self.store = store

    def retrieve(self, question: str) -> RetrievalResult:
        events = self.store.load()
        scoped_events, scope = self._time_scope(question, events)
        objects = self._current_objects(events)
        relationships = self._current_relationships(events)
        entities = self._entities(question, events)
        lowered = question.lower()
        spatial = any(term in lowered for term in ("left of", "right of", "above", "below", "near", "far"))
        changes = any(term in lowered for term in ("changed", "happened", "unusual", "what happened"))
        factual = any(term in lowered for term in ("did ", "has ", "when did", "picked up", "put down", "opened", "closed", "interacted", "used"))

        relevant_events = [event for event in scoped_events if self._matches(event, entities)]
        if spatial:
            relevant_relationships = [
                relation for relation in relationships
                if self._matches(relation, entities)
                and (not self._relation_requested(lowered) or relation["relation"] == self._relation_requested(lowered))
            ]
            relevant_events = [event for event in scoped_events if event.get("event_type", "").startswith("RELATIONSHIP_") and self._matches(event, entities)]
        elif not (changes or factual):
            relevant_events = [event for event in scoped_events if self._matches(event, entities)]
            relevant_relationships = [relation for relation in relationships if self._matches(relation, entities)]
        else:
            relevant_relationships = [relation for relation in relationships if self._matches(relation, entities)]

        relevant_objects = [obj for obj in objects if self._matches(obj, entities)] if entities else objects
        if not relevant_events and not relevant_relationships and entities:
            relevant_events = [event for event in scoped_events if self._mentions(event, entities)]
        return RetrievalResult(
            events=relevant_events[-80:],
            objects=relevant_objects[:40],
            relationships=relevant_relationships[:80],
            time_description=scope,
        )

    @staticmethod
    def _time_scope(question: str, events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        lowered = question.lower()
        now = datetime.now(timezone.utc)
        match = re.search(r"last\s+(\d+(?:\.\d+)?)\s*(second|minute|hour)s?", lowered)
        if match:
            amount = float(match.group(1))
            unit = match.group(2)
            seconds = amount * {"second": 1, "minute": 60, "hour": 3600}[unit]
            cutoff = now - timedelta(seconds=seconds)
            return [event for event in events if EventRetriever._timestamp(event) >= cutoff], f"last {match.group(1)} {unit}(s)"
        if "today" in lowered:
            start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            return [event for event in events if EventRetriever._timestamp(event) >= start], "today"
        before = re.search(r"before\s+(.+?)(?:\s+happened|\?|$)", lowered)
        if before:
            needle = before.group(1).strip()
            matching = [event for event in events if needle in json.dumps(event).lower()]
            if matching:
                end = EventRetriever._timestamp(matching[0])
                return [event for event in events if EventRetriever._timestamp(event) < end], f"before {needle}"
        return events, "available history"

    @staticmethod
    def _timestamp(event: dict[str, Any]) -> datetime:
        try:
            return datetime.fromisoformat(event["timestamp"])
        except (KeyError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)

    @staticmethod
    def _current_objects(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        latest: dict[Any, dict[str, Any]] = {}
        for event in events:
            event_type = event.get("event_type")
            object_id = event.get("object_id")
            if object_id is None:
                continue
            if event_type == "detection":
                latest[object_id] = event
            elif event_type == "OBJECT_APPEARED":
                latest[object_id] = {
                    "object_id": object_id,
                    "class": event.get("object_class"),
                    "label_class": event.get("object_class"),
                    "center": event.get("current_position"),
                    "bounding_box": event.get("current_bounding_box"),
                    "confidence": event.get("confidence"),
                    "timestamp": event.get("timestamp"),
                }
            elif event_type in {"OBJECT_MOVED", "OBJECT_CLASS_CHANGED"}:
                current = latest.get(object_id, {"object_id": object_id})
                current.update({
                    "class": event.get("object_class", current.get("class")),
                    "label_class": event.get("object_class", current.get("label_class")),
                    "center": event.get("current_position", current.get("center")),
                    "bounding_box": event.get("current_bounding_box", current.get("bounding_box")),
                    "confidence": event.get("confidence", current.get("confidence")),
                    "timestamp": event.get("timestamp"),
                })
                latest[object_id] = current
            elif event_type == "OBJECT_DISAPPEARED":
                latest.pop(event.get("object_id"), None)
        return sorted(latest.values(), key=lambda item: str(item.get("object_id")))

    @staticmethod
    def _current_relationships(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        active: dict[tuple[Any, Any, str], dict[str, Any]] = {}
        for event in events:
            if not event.get("event_type", "").startswith("RELATIONSHIP_"):
                continue
            key = (event.get("subject_id"), event.get("object_id"), event.get("relation"))
            if event["event_type"] in {"RELATIONSHIP_CREATED", "RELATIONSHIP_CHANGED"}:
                active[key] = event
            elif event["event_type"] == "RELATIONSHIP_REMOVED":
                active.pop(key, None)
        return list(active.values())

    @staticmethod
    def _entities(question: str, events: list[dict[str, Any]]) -> set[str]:
        haystack = json.dumps(events).lower()
        candidates: set[str] = set()
        lowered_question = question.lower()
        for canonical, aliases in EventRetriever.ENTITY_ALIASES.items():
            if any(alias in lowered_question for alias in aliases):
                candidates.add(canonical)
                candidates.update(aliases)
        for event in events:
            for field in ("object_id", "subject_id", "object_class", "subject_class", "class"):
                value = event.get(field)
                if value is not None and str(value).lower() in lowered_question:
                    candidates.add(str(value).lower())
        stop_words = {
            "what", "where", "when", "did", "the", "last", "seconds", "today", "before",
            "left", "right", "above", "below", "near", "far", "is", "to", "of", "a", "an",
        }
        known_terms = set(EventRetriever.ENTITY_ALIASES)
        known_terms.update(
            str(event.get(field)).lower()
            for event in events
            for field in ("object_class", "subject_class", "class")
            if event.get(field) is not None
        )
        query_words = re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)?\b", lowered_question)
        for value in query_words:
            if value in haystack and value not in stop_words:
                candidates.add(value)
            if value not in stop_words:
                close_match = get_close_matches(value, known_terms, n=1, cutoff=0.82)
                if close_match:
                    matched = close_match[0]
                    candidates.add(matched)
                    candidates.update(EventRetriever.ENTITY_ALIASES.get(matched, {matched}))
        return candidates

    @staticmethod
    def _mentions(record: dict[str, Any], entities: set[str]) -> bool:
        text = json.dumps(record).lower()
        return not entities or any(entity in text for entity in entities)

    @classmethod
    def _matches(cls, record: dict[str, Any], entities: set[str]) -> bool:
        return cls._mentions(record, entities)

    @staticmethod
    def _relation_requested(question: str) -> str | None:
        for relation in ("LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "NEAR", "FAR"):
            if relation.lower().replace("_", " ") in question:
                return relation
        return None


class GroqQueryResponder:
    """Ask Groq about retrieved evidence only, never the complete event log."""

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        self.model = model
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=api_key)

    def answer(self, question: str, evidence: dict[str, Any]) -> str:
        prompt = (
            "Answer the user's question using ONLY the retrieved JSON evidence. "
            "Do not guess or add facts. If evidence is insufficient, say so explicitly. "
            "Prefer deterministic relationships for spatial questions. Be concise and include "
            "confidence and timestamps when relevant. Return natural-language text only.\n"
            f"Question: {question}\nEvidence:\n{json.dumps(evidence, ensure_ascii=True)}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions about a vision event history.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


class DeterministicResponder:
    """Useful no-API fallback that still answers only from retrieved evidence."""

    def answer(self, question: str, evidence: dict[str, Any]) -> str:
        lowered = question.lower()
        if "how many" in lowered:
            requested = re.search(
                r"how many\s+([a-z][a-z ]*?)(?:\s+(?:are|were|is|have|currently|present|there)|\?|$)",
                lowered,
            )
            requested_class = "" if requested is None else requested.group(1).strip().rstrip("?")
            requested_class = {"people": "person", "persons": "person"}.get(requested_class, requested_class.rstrip("s"))
            known_classes = set(EventRetriever.ENTITY_ALIASES)
            for aliases in EventRetriever.ENTITY_ALIASES.values():
                known_classes.update(aliases)
            known_classes.update(
                str(obj.get("class", obj.get("label_class", ""))).lower()
                for obj in (
                    evidence["current_objects"]
                    + self._observed_objects(evidence["events"])
                )
                if obj.get("class", obj.get("label_class"))
            )
            close_class = get_close_matches(requested_class, known_classes, n=1, cutoff=0.75)
            if close_class:
                requested_class = close_class[0]
            requested_class = {
                "people": "person",
                "persons": "person",
                "human": "person",
                "humans": "person",
                "phone": "phone",
                "mobile": "phone",
                "cellphone": "phone",
            }.get(requested_class, requested_class)
            current_only = any(term in lowered for term in ("present", "currently", "now", "active"))
            candidates = evidence["current_objects"] if current_only else self._observed_objects(evidence["events"])
            matching = [
                obj for obj in evidence["current_objects"]
                if requested_class and requested_class in str(obj.get("class", obj.get("label_class", ""))).lower()
            ]
            if not current_only:
                matching = [
                    obj for obj in candidates
                    if requested_class and (
                        requested_class in str(obj.get("class", obj.get("label_class", ""))).lower()
                        or str(obj.get("class", obj.get("label_class", ""))).lower() in requested_class
                    )
                ]
            if requested_class:
                status = "currently supported by the recorded state" if current_only else "recorded in the event history"
                return f"There {('is' if len(matching) == 1 else 'are')} {len(matching)} {requested_class}{'' if len(matching) == 1 else 's'} {status}."
        if any(term in lowered for term in ("left of", "right of", "above", "below", "near", "far")):
            if evidence["current_relationships"]:
                relation = evidence["current_relationships"][0]
                return f"{relation.get('subject_class')} ({relation.get('subject_id')}) is {relation['relation']} {relation.get('object_class')} ({relation.get('object_id')})."
            return "The available relationship history is insufficient to answer that question."
        if any(term in lowered for term in ("did ", "has ", "picked up", "put down", "opened", "closed", "interacted", "used")):
            if evidence["events"]:
                event = evidence["events"][-1]
                return f"Yes. {event.get('event_type')} was recorded at {event.get('timestamp')} with confidence {event.get('confidence')}."
            return "No matching event was found in the available history."
        relationships = evidence["current_relationships"]
        if relationships:
            relation = relationships[0]
            return f"{relation.get('subject_class')} ({relation.get('subject_id')}) is {relation['relation']} {relation.get('object_class')} ({relation.get('object_id')})."
        if evidence["events"]:
            event = evidence["events"][-1]
            return f"The latest relevant event was {event.get('event_type')} at {event.get('timestamp')} with confidence {event.get('confidence')}."
        return "The available event history is insufficient to answer that question."

    @staticmethod
    def _observed_objects(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        observed: dict[Any, dict[str, Any]] = {}
        for event in events:
            object_id = event.get("object_id")
            object_class = event.get("object_class", event.get("class"))
            if object_id is not None and object_class:
                observed[object_id] = {
                    "object_id": object_id,
                    "class": object_class,
                    "label_class": object_class,
                }
            subject_id = event.get("subject_id")
            subject_class = event.get("subject_class")
            if subject_id is not None and subject_class:
                observed[subject_id] = {
                    "object_id": subject_id,
                    "class": subject_class,
                    "label_class": subject_class,
                }
        return list(observed.values())
