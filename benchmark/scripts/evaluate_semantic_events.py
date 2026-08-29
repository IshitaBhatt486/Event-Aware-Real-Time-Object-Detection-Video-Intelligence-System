"""Evaluate structured Gemini events without comparing descriptions."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import load_config, read_json, read_jsonl, resolve_path, timestamp_seconds, write_json, prf

def evaluate(predictions, truths, tolerance=2.0):
    used = set(); matched = []; latencies = []
    for prediction in predictions:
        candidates = []
        for index, truth in enumerate(truths):
            if index in used or prediction.get("event_type") != truth.get("event_type"): continue
            if any(prediction.get(field) is not None and truth.get(field) is not None and str(prediction[field]) != str(truth[field]) for field in ("subject_id", "object_id", "subject_class", "object_class")): continue
            distance = abs((timestamp_seconds(prediction.get("timestamp")) or 0) - (timestamp_seconds(truth.get("timestamp")) or 0))
            if distance <= tolerance: candidates.append((distance, index))
        if candidates:
            distance, index = min(candidates); used.add(index); matched.append(prediction); latencies.append(distance)
    result = prf(len(matched), len(predictions) - len(matched), len(truths) - len(matched)); confidences = [float(x["confidence"]) for x in predictions if isinstance(x.get("confidence"), (int, float))]
    result.update({"average_gemini_confidence": sum(confidences) / len(confidences) if confidences else None, "semantic_event_latency_seconds": sum(latencies) / len(latencies) if latencies else None, "gemini_event_count": len(predictions), "matched_events": len(matched)})
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions"); parser.add_argument("--ground-truth"); parser.add_argument("--config", default="benchmark/config.yaml"); parser.add_argument("--output")
    args = parser.parse_args(); config = load_config(args.config); base = Path(args.config).resolve().parents[1]; pp = resolve_path(args.predictions or config["paths"].get("raw_measurements"), base); tp = resolve_path(args.ground_truth or config["paths"]["semantic_ground_truth"], base)
    preds = read_jsonl(pp) if pp and pp.suffix == ".jsonl" else read_json(pp); refs = read_json(tp).get("events", []) if tp and tp.exists() else []; result = evaluate([x for x in preds if x.get("source") == "gemini" or x.get("event_source") == "gemini"], refs, config["evaluation"].get("semantic_tolerance_seconds", 2.0)); write_json(args.output or base / "results" / "semantic_metrics.json", result); print(result)
if __name__ == "__main__": main()
