"""Shared utilities for the isolated benchmark tools."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml
    except ImportError as error:
        raise SystemExit("Install benchmark/requirements.txt to read YAML configuration") from error
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data.setdefault("paths", {})
    data.setdefault("evaluation", {})
    return data


def resolve_path(value: str | Path | None, base: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else (base or project_root()) / path


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if isinstance(value, dict):
                records.append(value)
    return records


def timestamp_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def interval_start_end(record: dict[str, Any]) -> tuple[float, float]:
    start = timestamp_seconds(record.get("start_time", record.get("timestamp", record.get("start"))))
    end = timestamp_seconds(record.get("end_time", record.get("end", record.get("timestamp"))))
    return (start if start is not None else 0.0, end if end is not None else (start or 0.0))


def iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(item) for item in first]
    bx1, by1, bx2, by2 = [float(item) for item in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_records(predictions: list[dict[str, Any]], truths: list[dict[str, Any]], compatible, threshold: float = 0.0):
    """Greedily match each prediction to its best unused compatible truth."""
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            score = compatible(prediction, truth)
            if score >= threshold:
                candidates.append((score, pred_index, truth_index))
    matches: list[tuple[int, int, float]] = []
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    for score, pred_index, truth_index in sorted(candidates, reverse=True):
        if pred_index not in used_predictions and truth_index not in used_truths:
            used_predictions.add(pred_index)
            used_truths.add(truth_index)
            matches.append((pred_index, truth_index, score))
    return matches, set(range(len(predictions))) - used_predictions, set(range(len(truths))) - used_truths


def prf(true_positives: int, false_positives: int, false_negatives: int) -> dict[str, float]:
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def flatten_csv(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section, values in metrics.items():
        if isinstance(values, dict):
            for name, value in values.items():
                if not isinstance(value, (dict, list)):
                    rows.append({"section": section, "metric": name, "value": value})
    return rows


def write_metrics_csv(path: str | Path, metrics: dict[str, Any]) -> None:
    rows = flatten_csv(metrics)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
