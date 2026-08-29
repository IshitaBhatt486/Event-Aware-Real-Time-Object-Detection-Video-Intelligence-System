"""Run the existing local perception pipeline on configured benchmark videos."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import cv2
from common import load_config, project_root, read_json, resolve_path, write_json, write_metrics_csv

def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_report(path: Path, metrics: dict[str, Any], config: dict[str, Any]) -> None:
    lines = ["# Benchmark report", "", f"Measured at: {metrics.get('benchmark', {}).get('measured_at', 'unknown')}", "", "## Dataset and configuration", "", f"- Dataset: {config.get('benchmark', {}).get('dataset_name', 'unknown')} ({config.get('benchmark', {}).get('dataset_type', 'unknown')})", f"- Description: {config.get('benchmark', {}).get('description', '')}", f"- Videos processed: {metrics.get('benchmark', {}).get('videos', 0)}", "", "## Metrics", ""]
    for section, values in metrics.items():
        if section in {"benchmark", "limitations"} or not isinstance(values, dict): continue
        lines.append(f"### {section.replace('_', ' ').title()}")
        for name, value in values.items():
            if isinstance(value, (dict, list)): continue
            lines.append(f"- {name}: {value if value is not None else 'UNAVAILABLE (missing data or annotations)'}")
        lines.append("")
    lines += ["## Limitations", ""] + [f"- {item}" for item in metrics.get("limitations", [])]
    if not metrics.get("limitations"): lines.append("- None recorded by the runner; inspect dataset coverage before interpreting scores.")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")

class CaptureSink:
    def __init__(self): self.records = []
    def append(self, event): self.records.append(dict(event))

def run_video(video_path: Path, config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    root = project_root(); sys.path.insert(0, str(root))
    from object_detector import YOLOTracker, FrameEventDetector, detect_local_objects
    from relationship_detection import RelationshipDetector
    from temporal_reasoning import TemporalReasoner
    from ultralytics import YOLO
    sink = CaptureSink(); model = YOLO(str(resolve_path(config["model"], root)))
    tracker = YOLOTracker(model); detector = FrameEventDetector(sink, config["relationship"].get("movement_threshold", 20.0))
    relationships = RelationshipDetector(sink, **{key: config["relationship"][key] for key in ("near_threshold", "far_threshold", "directional_margin")})
    temporal = TemporalReasoner(sink, time_window=config["temporal"].get("window_seconds", 5.0), movement_threshold=config["relationship"].get("movement_threshold", 20.0), carry_distance=config["relationship"].get("near_threshold", 100.0), stationary_frames=config["temporal"].get("stationary_frames", 2))
    capture = cv2.VideoCapture(str(video_path)); if_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0; frame_id = 0; measurements = []; started = time.perf_counter(); first_wall = None
    while True:
        ok, frame = capture.read()
        if not ok: break
        frame_id += 1; frame_time = frame_id / if_fps if if_fps else time.perf_counter() - started; captured = datetime.now(timezone.utc).isoformat(); first_wall = first_wall or time.perf_counter()
        yolo_start = time.perf_counter(); detections = detect_local_objects(tracker, frame); yolo_end = time.perf_counter()
        yolo_completed = stamp()
        for item in detections: item.update({"frame_id": frame_id, "frame_time": frame_time})
        tracking_completed = stamp()
        raw_start = time.perf_counter(); raw_events = detector.update(detections); relationship_events = relationships.update(detections); temporal_events = temporal.update(detections, raw_events + relationship_events); local_end = time.perf_counter()
        relationship_completed = stamp()
        events = raw_events + relationship_events + temporal_events
        for event in events:
            event["frame_id"] = frame_id
            event["frame_time"] = frame_time
        record = {"frame_id": frame_id, "frame_time": frame_time, "frame_captured": captured, "yolo_completed": yolo_completed, "tracking_completed": tracking_completed, "relationship_processing_completed": relationship_completed, "event_persisted": stamp(), "detections": detections, "events": events, "source_video": str(video_path)}
        # Keep event-level rows in the same JSONL so evaluators can consume one artifact.
        measurements.append(record)
    capture.release(); duration = frame_id / if_fps if if_fps else (time.perf_counter() - started)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in measurements:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            for detection in record["detections"]: handle.write(json.dumps(detection, ensure_ascii=True) + "\n")
            for event in record["events"]: handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    return {"video": str(video_path), "frames": frame_id, "duration_seconds": duration, "processed_fps": frame_id / duration if duration else None, "events": sum(len(x["events"]) for x in measurements)}

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default="benchmark/config.yaml"); args = parser.parse_args(); config_path = Path(args.config).resolve(); config = load_config(config_path); base = config_path.parent
    videos = [resolve_path(path, base.parent) for path in config.get("benchmark_videos", [])]; videos = [path for path in videos if path and path.exists()]
    raw_dir = resolve_path(config["paths"]["raw_output"], base.parent); summaries = []
    for video in videos: summaries.append(run_video(video, config, raw_dir / f"{video.stem}.jsonl"))
    write_json(raw_dir / "run_summary.json", {"videos": summaries, "config": str(config_path), "measured_at": datetime.now(timezone.utc).isoformat()})
    metrics = {"benchmark": {"videos": len(videos), "frames": sum(x["frames"] for x in summaries), "measured_at": datetime.now(timezone.utc).isoformat()}, "limitations": []}
    combined = raw_dir / "all_measurements.jsonl"
    with combined.open("w", encoding="utf-8") as handle:
        for path in raw_dir.glob("*.jsonl"):
            if path != combined: handle.write(path.read_text(encoding="utf-8"))
    for name, section in (("evaluate_detection.py", "detection"), ("evaluate_tracking.py", "tracking"), ("evaluate_relationships.py", "relationships"), ("evaluate_temporal_events.py", "temporal_events"), ("evaluate_semantic_events.py", "semantic_events"), ("evaluate_anomalies.py", "anomalies")) if videos else ():
        output = base / "results" / f"_{section}.json"
        completed = subprocess.run([sys.executable, str(Path(__file__).with_name(name)), "--config", str(config_path), "--predictions", str(combined), "--output", str(output)], cwd=root, capture_output=True, text=True)
        if completed.returncode == 0 and output.exists(): metrics[section] = read_json(output)
        else: metrics["limitations"].append(f"{section} evaluator unavailable: {completed.stderr.strip() or 'missing ground truth'}")
    latency_output = base / "results" / "_latency.json"
    if videos:
        subprocess.run([sys.executable, str(Path(__file__).with_name("calculate_latency_throughput.py")), "--config", str(config_path), "--input", str(combined), "--output", str(latency_output)], cwd=root, check=False)
        if latency_output.exists(): metrics.update(read_json(latency_output))
    else:
        metrics["limitations"].append("No benchmark videos configured or found; no measured metrics were produced.")
        calls = metrics.get("throughput", {}).get("gemini_calls_per_hour")
        metrics["gemini"] = {"baseline_interval_seconds": config.get("gemini", {}).get("baseline_interval_seconds", 12.0), "event_driven_calls": calls, "baseline_calls": None, "call_reduction_percent": None}
        for section in ("detection", "tracking", "relationships", "temporal_events", "semantic_events", "anomalies"):
            metrics.setdefault(section, {"status": "unavailable"})
        metrics.setdefault("latency_ms", {"status": "unavailable"})
        metrics.setdefault("throughput", {"status": "unavailable"})
        write_report(base / "reports" / "benchmark_report.md", metrics, config)
    write_json(base / "results" / "metrics.json", metrics); write_metrics_csv(base / "results" / "metrics.csv", metrics)
    print(json.dumps({"videos": len(videos), "raw_output": str(raw_dir), "metrics": str(base / "results" / "metrics.json")}, indent=2))
if __name__ == "__main__": main()
