# Project guide: Event-Aware Object Detection

## What it does

This project turns a webcam stream into a queryable event history. It performs local object detection and persistent tracking, identifies state and spatial changes, derives a small set of temporal actions, optionally asks Gemini to interpret ambiguous interactions, and appends every emitted event to `events.jsonl`. A command-line query tool retrieves relevant evidence and answers questions from it.

It is a functional prototype and evaluation project, not a production surveillance system. The local loop, event pipeline, persistence, query layer, and isolated benchmarks are present. Production readiness still needs automated tests for the core modules, pinned dependencies, camera/input recovery, privacy controls, observability, data-retention controls, and a deployment/security review.

## Architecture

```text
webcam frame
  -> YOLO11 + ByteTrack (local detection and stable IDs)
  -> frame events (appeared, moved, disappeared, class changed)
  -> relationship rules (left/right/above/below/near/far)
  -> temporal rules (entered, left, approached, moved away, picked up, placed down)
  -> JSONL event log ----------------------> query_events.py
  -> optional event-triggered Gemini ------> validated semantic events --^
  -> OpenCV annotated preview
```

`object_detector.py` is the orchestration entry point. `relationship_detection.py` evaluates every active tracked-object pair. `temporal_reasoning.py` keeps a bounded recent history and emits only rule-supported higher-level actions. `gemini_event_trigger.py` runs an asynchronous worker, sends a frame only after a meaningful event or ambiguity and cooldown, then validates and deduplicates Gemini output. `event_storage.py` appends durable JSONL records. `event_query.py` retrieves a time- and entity-scoped subset of those records before responding.

## End-to-end walkthrough

1. `object_detector.py` loads YOLO weights, opens the selected webcam, and starts Gemini only when a key is present.
2. Each frame goes through YOLO tracking with persistent ByteTrack IDs. Detections below 0.4 confidence are ignored.
3. The frame detector emits appearance/disappearance and movement events when displacement exceeds `--movement-threshold`.
4. Relationship rules emit transitions for direction and distance, rather than repeat the same fact every frame.
5. Temporal rules correlate a bounded history to infer entry/exit, approach/away, movement, and conservative pick-up/put-down actions.
6. Events are retained in memory and written immediately as JSONL, so the query CLI can run independently.
7. When triggered, the background Gemini worker receives a copied frame plus bounded context. Its schema-constrained answer is validated for allowed event types, confidence, IDs, evidence, and duplicates before saving.
8. OpenCV draws local boxes. On `q`, the worker and event file close and the camera is released.
9. `query_events.py` scopes event evidence by time, entity, and spatial intent, then answers via Groq or an offline deterministic fallback.

## Tech stack and decisions

| Technology | Role | Why it was chosen |
| --- | --- | --- |
| Python | application language | Readable prototype surface and mature CV/AI libraries. |
| Ultralytics YOLO11 | local detection | Fast on-device detections work without API/network access. |
| ByteTrack | persistent IDs | Enables event, relationship, and temporal reasoning across frames. |
| OpenCV | capture/display | Direct webcam access and low-latency annotation. |
| Rule-based geometry | spatial/temporal events | Deterministic, inspectable, inexpensive results. |
| Gemini | optional semantics | Used only where geometry is insufficient; cooldown controls cost/latency. |
| JSONL | event store | Append-only, inspectable, simple for a prototype. |
| Groq | optional answers | Converts retrieved evidence to concise natural-language answers. |
| Benchmark scripts | evaluation | Isolate measurement from the live webcam pipeline. |

## Configuration and operation

Use `python object_detector.py --help` for all controls. Key options are `--camera`, `--yolo-model`, `--no-gemini`, movement/distance thresholds, temporal settings, and Gemini cooldown/confidence controls. Keep API keys in environment variables or ignored `.env`; never commit them. Local-only mode is the recommended first run.

Run the isolated benchmark suite after installing `benchmark/requirements.txt`, configuring `benchmark/config.yaml`, and adding approved videos/labels:

```powershell
python benchmark/scripts/benchmark_pipeline.py --config benchmark/config.yaml
```

It writes raw records, JSON/CSV metrics, and a Markdown report under `benchmark/`. It does not alter the live pipeline.

## Challenges and resolutions

| Challenge | Resolution | Remaining limitation |
| --- | --- | --- |
| API latency/cost | YOLO runs locally every frame; Gemini is asynchronous, event-triggered, and cooled down. | Semantic result may arrive after the displayed frame. |
| Object identity | Persistent ByteTrack IDs key historical state. | IDs can switch under occlusion/crowding. |
| Geometric flicker | Margins and transition-only events reduce noise. | Thresholds need camera-specific tuning. |
| Uncertain actions | Rules use bounded evidence; Gemini output is allow-listed and validated. | A pick-up is a heuristic, not proof of contact. |
| LLM hallucination | Retrieval scopes evidence; Groq is told to use only it; offline mode is deterministic. | Intent parsing is intentionally basic. |
| Durable history | Flushed JSONL supports independent querying. | Unsuitable for concurrent writers/high volume. |

## Improvement roadmap

1. Add core unit/integration tests and run them in CI.
2. Pin dependencies and document supported OS/GPU combinations.
3. Move JSONL to SQLite/PostgreSQL for indexing, retention, concurrency, and deletion.
4. Add zones, calibration, per-class thresholds, sampling, and tracker-reset handling.
5. Add structured logs, health metrics, retry/backoff, input recovery, and configurable model/provider versions.
6. Before sensitive deployment, add review UI, access control, encryption, retention, consent, and audit logs.
7. Validate on representative labeled footage and report detection/tracking/event quality plus latency/throughput.

## Interview questions

1. **Why not send every frame to Gemini?** Local YOLO is faster, cheaper, and resilient offline. Gemini is an additive semantic layer for meaningful changes.
2. **How are objects tracked?** ByteTrack’s persistent ID links detections across frames and keys the history.
3. **How are repeated events avoided?** The system emits state transitions, retains prior relationships, bounds history, applies cooldowns, and deduplicates semantic event keys.
4. **What constitutes `PICKED_UP`?** A stationary non-person object moves near a moving person with aligned movement. It remains an inference with recorded evidence.
5. **Why JSONL rather than a database?** It is minimal and inspectable for a prototype; a database is the appropriate next step for scale.
6. **How is LLM output controlled?** A response schema, allow-list, confidence cutoff, ID/class checks, evidence requirement, and duplicate suppression run before persistence.
7. **What accuracy risks remain?** Detector misses, ID switches, 2D distance, occlusion, lighting, and semantic-model errors.
8. **How would you evaluate it?** Labeled videos plus detection, tracking, event, semantic, latency, and throughput metrics in the isolated benchmark suite.
9. **What privacy issues matter?** Camera data requires purpose limitation, notice/consent where applicable, minimization, access control, retention limits, and secure storage.
10. **How would you scale it?** Decouple capture/inference/eventing with queues, use a database, accelerate inference, and monitor worker health.

## Documentation maintenance contract

For every functional addition or behavior change, update this guide in the same change set:

- revise architecture/walkthrough when control or data flow changes;
- revise setup/configuration and user-facing capabilities as needed;
- record the decision, trade-off, limitation, and validation;
- add or revise an interview question when it improves design discussion.

The pull-request template and workflow require a documentation change alongside Python changes. For local/non-PR work, follow [CONTRIBUTING.md](CONTRIBUTING.md). Repository administrators must mark the `Documentation contract` workflow as a required protected-branch check.
