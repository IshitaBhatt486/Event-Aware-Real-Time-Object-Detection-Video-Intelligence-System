# Benchmark datasets

Place custom videos in a location listed by `benchmark_videos` in `benchmark/config.yaml`. Keep annotations in `ground_truth/`; timestamps may be video seconds or ISO-8601 values, and IDs must remain stable throughout a video.

Public options are configuration-only: MOT17/MOT20 provide MOT tracking annotations; VisDrone provides detection/tracking annotations; COCO is suitable for frame detection evaluation. Download and license these datasets separately, then point the config paths at converted JSON files. The benchmark does not download large datasets. Relationships, temporal reasoning, Gemini semantic events, and anomalies generally require custom recorded videos with manual labels.

The example files in `ground_truth/` are intentionally tiny schemas, not benchmark results. Copy them per video and add `video_id` when combining test sets.
