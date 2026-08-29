"""Evaluate frame detections from benchmark JSONL against box annotations."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from common import iou, load_config, match_records, read_jsonl, read_json, prf, resolve_path, write_json


def _frames(value):
    if isinstance(value, list): return value
    if isinstance(value, dict): return value.get("frames", value.get("annotations", []))
    return []


def evaluate(predictions, truths, iou_threshold=0.5, confidence_threshold=0.0):
    predictions = [p for p in predictions if float(p.get("confidence", 1)) >= confidence_threshold]
    grouped_predictions = defaultdict(list); grouped_truths = defaultdict(list)
    for item in predictions: grouped_predictions[item.get("frame_id", item.get("frame", 0))].append(item)
    for item in truths: grouped_truths[item.get("frame_id", item.get("frame", 0))].append(item)
    tp = fp = fn = 0; matched_ious = []; per_class = defaultdict(lambda: [0, 0, 0])
    for frame_id in set(grouped_predictions) | set(grouped_truths):
        preds, refs = grouped_predictions[frame_id], grouped_truths[frame_id]
        def compatible(pred, truth):
            if pred.get("class", pred.get("label_class")) != truth.get("class", truth.get("label_class")): return -1
            return iou(pred.get("bbox", pred.get("bounding_box")), truth.get("bbox", truth.get("bounding_box")))
        matches, unmatched_p, unmatched_t = match_records(preds, refs, compatible, iou_threshold)
        tp += len(matches); fp += len(unmatched_p); fn += len(unmatched_t)
        matched_ious.extend(score for _, _, score in matches)
        for p, t, _ in matches: per_class[preds[p].get("class", preds[p].get("label_class"))][0] += 1
        for p in unmatched_p: per_class[preds[p].get("class", preds[p].get("label_class"))][1] += 1
        for t in unmatched_t: per_class[refs[t].get("class", refs[t].get("label_class"))][2] += 1
    result = prf(tp, fp, fn)
    result.update({"true_positives": tp, "false_positives": fp, "false_negatives": fn,
                   "mean_iou_matched": sum(matched_ious) / len(matched_ious) if matched_ious else None,
                   "mAP50": None, "mAP50_95": None,
                   "per_class": {k: dict(zip(("tp", "fp", "fn"), v)) for k, v in per_class.items()}})
    # AP requires confidence-ranked predictions; calculate a compact class/frame AP when boxes are available.
    result["mAP50"] = _map_at_iou(predictions, truths, 0.5)
    result["mAP50_95"] = sum((_map_at_iou(predictions, truths, threshold) or 0) for threshold in [x / 100 for x in range(50, 100, 5)]) / 10 if truths else None
    return result


def _map_at_iou(predictions, truths, threshold):
    classes = sorted({p.get("class", p.get("label_class")) for p in predictions} | {t.get("class", t.get("label_class")) for t in truths})
    average_precisions = []
    for class_name in classes:
        preds = sorted([p for p in predictions if p.get("class", p.get("label_class")) == class_name], key=lambda x: float(x.get("confidence", 0)), reverse=True)
        refs = [t for t in truths if t.get("class", t.get("label_class")) == class_name]
        used = set(); tp = []; fp = []
        for pred in preds:
            candidates = [(iou(pred.get("bbox", pred.get("bounding_box")), ref.get("bbox", ref.get("bounding_box"))), index) for index, ref in enumerate(refs) if index not in used and pred.get("frame_id", pred.get("frame", 0)) == ref.get("frame_id", ref.get("frame", 0))]
            score, index = max(candidates, default=(0, -1))
            if score >= threshold: used.add(index); tp.append(1); fp.append(0)
            else: tp.append(0); fp.append(1)
        if refs:
            cumulative_tp = 0; cumulative_fp = 0; precisions = [0.0]
            for true_positive, false_positive in zip(tp, fp):
                cumulative_tp += true_positive; cumulative_fp += false_positive
                precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
            average_precisions.append(max(precisions) * (len(used) / len(refs)) if precisions else 0.0)
    return sum(average_precisions) / len(average_precisions) if average_precisions else None


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions"); parser.add_argument("--ground-truth"); parser.add_argument("--config", default="benchmark/config.yaml"); parser.add_argument("--output")
    args = parser.parse_args(); config = load_config(args.config); base = Path(args.config).resolve().parents[1]
    prediction_path = resolve_path(args.predictions or config["paths"].get("raw_measurements"), base); truth_path = resolve_path(args.ground_truth or config["paths"]["detection_ground_truth"], base)
    predictions = read_jsonl(prediction_path) if prediction_path and prediction_path.suffix == ".jsonl" else _frames(read_json(prediction_path))
    predictions = [item for item in predictions if item.get("bbox", item.get("bounding_box")) is not None]
    truths = _frames(read_json(truth_path)) if truth_path and truth_path.exists() else []
    result = evaluate(predictions, truths, config["evaluation"].get("iou_threshold", .5), config["evaluation"].get("confidence_threshold", .4))
    write_json(args.output or base / "results" / "detection_metrics.json", result); print(result)
if __name__ == "__main__": main()
