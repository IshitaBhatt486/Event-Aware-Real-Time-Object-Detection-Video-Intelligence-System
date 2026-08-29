# Benchmark framework

This isolated framework evaluates the existing YOLO + ByteTrack + event/relationship/temporal pipeline. It does not change the real-time camera pipeline. Add videos to `benchmark_videos` and manually labeled JSON to `datasets/ground_truth/`, then install `benchmark/requirements.txt` and run:

```powershell
python benchmark/scripts/benchmark_pipeline.py --config benchmark/config.yaml
```

Raw per-frame and event measurements are written to `results/raw/`. Aggregated machine-readable output is written to `results/metrics.json` and `results/metrics.csv`; the human report is `reports/benchmark_report.md`. Individual evaluators accept `--predictions`, `--ground-truth`, `--config`, and `--output`, for example:

```powershell
python benchmark/scripts/evaluate_detection.py --config benchmark/config.yaml
```

Detection uses one-to-one IoU matching and AP summaries. Tracking uses CLEAR-MOT-style counts and identity consistency. Relationship and event evaluators match stable IDs; temporal, semantic, and anomaly matches use configurable tolerances. Gemini descriptions are never compared as strings. Missing annotations produce unavailable (`null`) values and a limitation entry. Gemini call reduction requires measured baseline and event-driven call counts in the raw records; no calls or results are fabricated.

For public datasets and annotation conventions, see `datasets/README.md`. The example ground truth files are schemas only.
