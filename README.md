# Gemini Object Detector

Detect objects from a live webcam feed with local YOLO and optionally enrich detections with Gemini, then display the result with OpenCV.

## Setup

Python 3.9 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a Gemini API key in Google AI Studio, then set it in the environment. The key is never stored in the project:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

On macOS or Linux:

```bash
export GEMINI_API_KEY="your-api-key"
```

## Run

```powershell
python object_detector.py
```

The camera preview opens in an OpenCV window. Press `q` in that window to quit. YOLO detects locally on every frame, so boxes update immediately without using API quota. Gemini runs in the background once per minute by default. Download `yolo11n.pt` once from [Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt) and place it in this project folder before the first run.

```powershell
python object_detector.py
```

To use no Gemini quota at all, run local-only mode:

```powershell
python object_detector.py --no-gemini
```

To change the optional Gemini interval:

```powershell
python object_detector.py --interval 60
```

The free tier may return a `429 RESOURCE_EXHAUSTED` quota error if the interval is too short. The local YOLO detections continue and the program does not crash.

For a different camera device, pass its index:

```powershell
python object_detector.py --camera 1
```

Gemini still returns JSON containing `label` and `box_2d: [ymin, xmin, ymax, xmax]`; its 0-1000 coordinates are converted to pixels before drawing. YOLO's local labels are drawn immediately, while Gemini results are added when available. To use a weight file at another location, pass `--yolo-model path\to\yolo11n.pt`.
