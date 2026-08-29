"""Evaluate anomaly events against labeled anomalous intervals."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import interval_start_end, load_config, read_json, read_jsonl, resolve_path, timestamp_seconds, write_json, prf

def evaluate(predictions, truths, tolerance=2.0, duration_seconds=None):
    used = set(); latencies = []
    for prediction in predictions:
        predicted_time = timestamp_seconds(prediction.get("timestamp")) or interval_start_end(prediction)[0]
        candidates = []
        for index, truth in enumerate(truths):
            if index in used: continue
            start, end = interval_start_end(truth)
            if start - tolerance <= predicted_time <= end + tolerance: candidates.append((abs(predicted_time - start), index))
        if candidates:
            latency, index = min(candidates); used.add(index); latencies.append(latency)
    tp = len(used); fp = len(predictions) - tp; fn = len(truths) - tp; hours = (duration_seconds or 0) / 3600
    result = prf(tp, fp, fn); result.update({"false_positive_rate": fp / (fp + tp) if fp + tp else 0.0, "false_alerts_per_hour": fp / hours if hours else None, "detection_latency_seconds": sum(latencies) / len(latencies) if latencies else None})
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions"); parser.add_argument("--ground-truth"); parser.add_argument("--config", default="benchmark/config.yaml"); parser.add_argument("--output")
    args = parser.parse_args(); config = load_config(args.config); base = Path(args.config).resolve().parents[1]; pp = resolve_path(args.predictions or config["paths"].get("raw_measurements"), base); tp = resolve_path(args.ground_truth or config["paths"]["anomaly_ground_truth"], base)
    preds = read_jsonl(pp) if pp and pp.suffix == ".jsonl" else read_json(pp); refs = read_json(tp).get("anomalies", []) if tp and tp.exists() else []; result = evaluate([x for x in preds if x.get("event_type") == "ANOMALY" or x.get("anomaly") is True], refs, config["evaluation"].get("anomaly_tolerance_seconds", 2.0), config["benchmark"].get("duration_seconds")); write_json(args.output or base / "results" / "anomaly_metrics.json", result); print(result)
if __name__ == "__main__": main()
