"""Evaluate tracked detections with CLEAR-MOT-style and IDF1 metrics."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from common import iou, load_config, read_json, read_jsonl, resolve_path, write_json


def evaluate(predictions, truths, iou_threshold=.5):
    pred_by_frame = defaultdict(list); truth_by_frame = defaultdict(list)
    for item in predictions: pred_by_frame[item.get("frame_id", item.get("frame", 0))].append(item)
    for item in truths: truth_by_frame[item.get("frame_id", item.get("frame", 0))].append(item)
    tp = fp = fn = switches = 0; previous_matches = {}; pair_counts = defaultdict(int)
    for frame in sorted(set(pred_by_frame) | set(truth_by_frame), key=str):
        preds, refs = pred_by_frame[frame], truth_by_frame[frame]; candidates = []
        for pi, pred in enumerate(preds):
            for ti, truth in enumerate(refs):
                if pred.get("class", pred.get("label_class")) == truth.get("class", truth.get("label_class")):
                    candidates.append((iou(pred.get("bbox", pred.get("bounding_box")), truth.get("bbox", truth.get("bounding_box"))), pi, ti))
        used_p = set(); used_t = set()
        for score, pi, ti in sorted(candidates, reverse=True):
            if score < iou_threshold or pi in used_p or ti in used_t: continue
            used_p.add(pi); used_t.add(ti); tp += 1
            pred_id = str(preds[pi].get("track_id", preds[pi].get("object_id"))); truth_id = str(refs[ti].get("track_id", refs[ti].get("object_id")))
            pair_counts[(truth_id, pred_id)] += 1
            if truth_id in previous_matches and previous_matches[truth_id] != pred_id: switches += 1
            previous_matches[truth_id] = pred_id
        fp += len(preds) - len(used_p); fn += len(refs) - len(used_t)
    identity_true = sum(max((count for (truth_id, _), count in pair_counts.items() if truth_id == ref_id), default=0) for ref_id in {key[0] for key in pair_counts})
    idtp = sum(max((count for (truth_id, pred_id), count in pair_counts.items() if truth_id == ref_id), default=0) for ref_id in {key[0] for key in pair_counts})
    idfp = max(0, tp - idtp) + fp; idfn = max(0, tp - idtp) + fn
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn else 0.0
    return {"idf1": idf1, "id_switches": switches, "mota": 1 - (fn + fp + switches) / (tp + fn) if tp + fn else None,
            "tracking_precision": tp / (tp + fp) if tp + fp else 0.0, "tracking_recall": tp / (tp + fn) if tp + fn else 0.0,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn, "identity_true_positives": identity_true}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions"); parser.add_argument("--ground-truth"); parser.add_argument("--config", default="benchmark/config.yaml"); parser.add_argument("--output")
    args = parser.parse_args(); config = load_config(args.config); base = Path(args.config).resolve().parents[1]
    pred_path = resolve_path(args.predictions or config["paths"].get("raw_measurements"), base); truth_path = resolve_path(args.ground_truth or config["paths"]["tracking_ground_truth"], base)
    preds = read_jsonl(pred_path) if pred_path and pred_path.suffix == ".jsonl" else (read_json(pred_path) if pred_path else [])
    preds = [item for item in preds if item.get("bbox", item.get("bounding_box")) is not None]
    refs = read_json(truth_path).get("frames", []) if truth_path and truth_path.exists() else []
    result = evaluate(preds, refs, config["evaluation"].get("iou_threshold", .5)); write_json(args.output or base / "results" / "tracking_metrics.json", result); print(result)
if __name__ == "__main__": main()
