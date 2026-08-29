"""Evaluate labeled spatial relations by stable subject/object IDs."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from common import load_config, read_json, read_jsonl, resolve_path, write_json, prf
LABELS = {"LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "NEAR", "FAR"}

def evaluate(predictions, truths):
    def key(x): return (x.get("frame_id", x.get("frame", 0)), str(x.get("subject_id")), str(x.get("object_id")))
    pred = {key(x): x.get("relation") for x in predictions if x.get("relation") in LABELS}; ref = {key(x): x.get("relation") for x in truths if x.get("relation") in LABELS}
    tp = sum(pred.get(k) == v for k, v in ref.items()); fp = sum(k not in ref or pred[k] != v for k, v in pred.items()); fn = sum(k not in pred for k in ref)
    result = prf(tp, fp, fn); result["accuracy"] = tp / len(ref) if ref else None; result["support"] = len(ref)
    result["per_relation"] = {}
    for label in sorted(LABELS):
        p = [x for x in predictions if x.get("relation") == label]; t = [x for x in truths if x.get("relation") == label]; matched = sum(key(x) in ref and pred.get(key(x)) == label for x in p)
        result["per_relation"][label] = {**prf(matched, len(p) - matched, len(t) - matched), "support": len(t)}
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions"); parser.add_argument("--ground-truth"); parser.add_argument("--config", default="benchmark/config.yaml"); parser.add_argument("--output")
    args = parser.parse_args(); config = load_config(args.config); base = Path(args.config).resolve().parents[1]; pp = resolve_path(args.predictions or config["paths"].get("raw_measurements"), base); tp = resolve_path(args.ground_truth or config["paths"]["relationships_ground_truth"], base)
    predictions = read_jsonl(pp) if pp and pp.suffix == ".jsonl" else read_json(pp); truths = read_json(tp).get("relationships", []) if tp and tp.exists() else []
    result = evaluate([x for x in predictions if x.get("relation")], truths); write_json(args.output or base / "results" / "relationship_metrics.json", result); print(result)
if __name__ == "__main__": main()
