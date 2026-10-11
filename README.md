# Event-Aware Object Detection

Real-time webcam vision that detects and tracks objects locally, converts changes into structured events, and can optionally use Gemini to interpret higher-level interactions. Events are persisted as JSONL and can be queried in natural language.

## Quick start

Requires Python 3.9+, a webcam, and the `yolo11n.pt` weights file in the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python object_detector.py --no-gemini
```

Press `q` in the preview window to stop. Use another camera with `--camera 1`.

## Optional AI features

Gemini enriches significant local events; it is never required for local detection.

```powershell
$env:GEMINI_API_KEY = "your-api-key"
python object_detector.py --gemini-cooldown 10
```

Query saved events with Groq, or use the deterministic offline fallback:

```powershell
$env:GROQ_API_KEY = "your-api-key"
python query_events.py
python query_events.py --no-llm
```

## Project guide

The complete architecture, execution walkthrough, trade-offs, limitations, interview questions, benchmark usage, and improvement plan are in [PROJECT_GUIDE.md](PROJECT_GUIDE.md). Benchmark-specific instructions are in [benchmark/README.md](benchmark/README.md).
