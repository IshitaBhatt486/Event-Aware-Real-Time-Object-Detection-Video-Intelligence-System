"""Evaluate temporal events with type/ID matching and timestamp tolerance."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import interval_start_end, load_config, read_json, read_jsonl, resolve_path, timestamp_seconds, write_json, prf

def evaluate(predictions, truths, tolerance=2.0):
    used = set(); matches = []; latencies = []
    for prediction in predictions:
        best = None
        for index, truth in enumerate(truths):
            if index in used or prediction.get("event_type") != truth.get("event_type"): continue
            if any(prediction.get(field) is not None and truth.get(field) is not None and str(prediction[field]) != str(truth[field]) for field in ("subject_id", "object_id")): continue
            distance = abs((timestamp_seconds(prediction.get("timestamp")) or 0) - (timestamp_seconds(truth.get("timestamp")) or 0))
            if distance <= tolerance and (best is None or distance < best[0]): best = (distance, index)
        if best: used.add(best[1]); matches.append(prediction); latencies.append(best[0])
    result = prf(len(matches), len(predictions) - len(matches), len(truths) - len(matches)); result["matched_events"] = len(matches); result["mean_detection_latency_seconds"] = sum(latencies) / len(latencies) if latencies else None
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions"); parser.add_argument("--ground-truth"); parser.add_argument("--config", default="benchmark/config.yaml"); parser.add_argument("--output")
    args = parser.parse_args(); config = load_config(args.config); base = Path(args.config).resolve().parents[1]; pp = resolve_path(args.predictions or config["paths"].get("raw_measurements"), base); tp = resolve_path(args.ground_truth or config["paths"]["temporal_ground_truth"], base)
    preds = read_jsonl(pp) if pp and pp.suffix == ".jsonl" else read_json(pp); refs = read_json(tp).get("events", []) if tp and tp.exists() else []; allowed = set(config["evaluation"].get("temporal_event_types", [])); result = evaluate([x for x in preds if x.get("event_type") in allowed], refs, config["evaluation"].get("temporal_tolerance_seconds", 2.0)); write_json(args.output or base / "results" / "temporal_metrics.json", result); print(result)
if __name__ == "__main__": main()
