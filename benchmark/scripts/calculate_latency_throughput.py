"""Calculate stage latency percentiles and throughput from instrumented JSONL."""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter
from common import load_config, percentile, read_jsonl, resolve_path, timestamp_seconds, write_json
STAGES = {"yolo_latency_ms": ("frame_captured", "yolo_completed"), "tracking_latency_ms": ("yolo_completed", "tracking_completed"), "local_perception_latency_ms": ("frame_captured", "relationship_processing_completed"), "gemini_latency_ms": ("gemini_request_sent", "gemini_response_received"), "logging_latency_ms": ("gemini_response_received", "event_persisted"), "end_to_end_latency_ms": ("frame_captured", "event_persisted")}

def evaluate(records):
    values = {name: [] for name in STAGES}; frame_times = []; event_count = 0; gemini_calls = 0
    for record in records:
        if record.get("event_type") and record.get("event_type") != "detection": event_count += 1
        if record.get("gemini_request_sent") is not None: gemini_calls += 1
        captured = timestamp_seconds(record.get("frame_captured", record.get("timestamp"))); frame_times.append(captured)
        for name, (start, end) in STAGES.items():
            first = timestamp_seconds(record.get(start)); last = timestamp_seconds(record.get(end))
            if first is not None and last is not None and last >= first: values[name].append((last - first) * 1000)
    latency = {name: {"mean": sum(items) / len(items) if items else None, "p50": percentile(items, .5), "p95": percentile(items, .95), "p99": percentile(items, .99), "samples": len(items)} for name, items in values.items()}
    duration = max(frame_times) - min(frame_times) if len(frame_times) > 1 else 0
    input_fps = (len(frame_times) - 1) / duration if duration > 0 else None
    return {"latency_ms": latency, "throughput": {"input_fps": input_fps, "processed_fps": input_fps, "detections_per_second": None, "events_per_second": event_count / duration if duration else None, "gemini_calls_per_minute": gemini_calls / (duration / 60) if duration else None, "gemini_calls_per_hour": gemini_calls / (duration / 3600) if duration else None}}

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input"); parser.add_argument("--config", default="benchmark/config.yaml"); parser.add_argument("--output")
    args = parser.parse_args(); config = load_config(args.config); base = Path(args.config).resolve().parents[1]; path = resolve_path(args.input or config["paths"].get("raw_measurements"), base); result = evaluate(read_jsonl(path)); write_json(args.output or base / "results" / "latency_throughput.json", result); print(result)
if __name__ == "__main__": main()
