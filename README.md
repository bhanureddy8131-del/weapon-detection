# SmartVision Pro - Real-Time Threat Detection

SmartVision Pro is a computer vision monitoring platform. Detection and action-recognition capabilities depend on the weight files actually installed; the dashboard reports when it is using a general-purpose fallback model.

---

## 🚀 Key Features

- **Multi-Stage Detection Pipeline**: Integrates YOLO, ByteTrack, optional LSTM action recognition, and MediaPipe Pose estimation.
- **Object Recognition**: Classes are determined by the loaded YOLO weights. Firearm detection requires the project's trained six-class model; the general COCO fallback does not recognize firearms.
- **Action Recognition**: Fighting and shooting recognition require a compatible LSTM checkpoint and pose sequence data.
- **Alert System**: Sends real-time security alerts via Telegram and Email with attached snapshot captures.
- **Interactive Streamlit Dashboard**: Offers latest-frame webcam capture, incident history, configurable supported classes, confidence thresholds, and video-file input.
- **Low-Latency Webcam**: Captures continuously into a latest-frame slot and runs inference off the Streamlit UI thread; stale frames are discarded rather than queued. The capture worker retries camera disconnects, and the dashboard hides frames older than 500 ms rather than presenting them as live.
- **FastAPI Alert Server**: SQLite-backed backend serving alert data, static snapshot files, and REST API endpoints.

---

## ⚡ Quick Start

### 1. Prerequisites
- **Python 3.11** (recommended for the current MediaPipe Pose API)
- **Virtual Environment**: `.venv/` in this workspace, or create one locally.

### 2. Launch the Application

#### On Windows (PowerShell / Command Prompt):
```powershell
.\start.ps1
# or
start.bat
```

#### On Linux / macOS:
```bash
./start.sh
```

This launches both components simultaneously:
- **Backend Server (FastAPI)**: `http://localhost:8000`
- **Dashboard UI (Streamlit)**: `http://localhost:8501`

---

## ⚙️ Setup & Installation (Manual Setup)

```bash
# 1. Clone the repository
git clone https://github.com/Arunkoomath/wepondetection.git
cd wepondetection

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
# Windows:
pip install -r environment/requirements-windows.txt
# macOS:
pip install -r environment/requirements-macos.txt
```

---

## 🏗️ System Architecture

```
smart-vision/
├── data/
│   ├── input_videos/       # Input videos for processing
│   ├── incidents.db        # SQLite database storing security incidents
│   └── processed/          # Processed intermediate data
├── .venv/                  # Virtual Python environment
├── environment/            # Platform-specific requirements files
├── logs/                   # System and threat event logs
├── models/                 # Model files (YOLO, LSTM weights)
├── pipeline/
│   ├── app.py              # Streamlit Web Dashboard
│   └── run_inference.py    # Core Multi-Stage Inference Engine
├── server/
│   ├── main.py             # FastAPI Server & REST API endpoints
│   ├── models.py           # SQLAlchemy database schemas
│   ├── schemas.py          # Pydantic data validation schemas
│   └── services/           # Email & Telegram notification services
├── src/
│   ├── config.py           # Main configuration file (thresholds, options)
│   ├── modeling/           # YOLO detectors, LSTM action models, MediaPipe estimators
│   └── utils/              # Helper utilities
├── start.bat               # Windows Batch startup script
├── start.ps1               # Windows PowerShell startup script
├── start.sh                # Unix/Linux/macOS startup script
└── yolo11s.pt              # Fallback YOLO detection model
```

---

## 📡 API Endpoints (FastAPI Backend)

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Health check endpoint (`{"status": "running"}`) |
| `GET` | `/api/v1/incidents` | List recent incident alerts |
| `POST` | `/api/v1/incidents` | Create a new threat incident alert & trigger notifications |
| `GET` | `/static/alerts/{filename}` | Retrieve snapshot images attached to alert incidents |

---

## 🔧 Configuration Options (`src/config.py`)

- `CONFIDENCE_THRESHOLD`: Object detection confidence threshold (default: `0.10`).
- `ACTION_CONFIDENCE_THRESHOLD`: Minimum confidence for action recognition (default: `0.85`).
- `SMARTVISION_ACTION_CONFIDENCE_THRESHOLD`: Environment override for the action confidence threshold (0 to 1).
- `DRAW_PERSON_BOXES`: Enable bounding box overlays for detected persons.
- `DRAW_LABELS`: Enable text labels for detections.
- `API_BASE_URL`: Incident API base (`SMARTVISION_API_BASE_URL`, default `http://127.0.0.1:8000/api/v1`).
- `YOLO_IMAGE_SIZE`: Inference size (`SMARTVISION_YOLO_IMAGE_SIZE`, default `416`; the dashboard allows changing it).
- `YOLO_IOU_THRESHOLD`: YOLO NMS IoU threshold (`SMARTVISION_YOLO_IOU`, default `0.70`).
- `SMARTVISION_CONFIDENCE_THRESHOLD`, `SMARTVISION_PERSON_CONFIDENCE`, `SMARTVISION_GUN_CONFIDENCE`, and `SMARTVISION_KNIFE_CONFIDENCE`: Detection thresholds. The UI can raise, but not lower, the configured global threshold.

## Model files and supported classes

The custom detector path is `models/yolo11s_fine_tune_v3.pt`. The application checks its six class names, matching weights hash, precision/recall/mAP metrics, and representative validation-image coverage. The training script saves predictions for one validation image per class to a unique run folder. The COCO fallback's generic `knife` prediction is shown as detector-only and is never promoted to the project's validated `knife` threat class.

When the custom file is absent, the app explicitly falls back to the generic 80-class `yolo11s.pt` COCO model; that model has no `gun` or `atm` class. It can show its `person`, `backpack`, `handbag`, and generic `knife` predictions, but the generic knife is not treated as a project weapon.

Production alerts stay disabled until an operator has reviewed the validation metrics and saved representative images. After that review, set `production_alerts_approved` to `true` in `models/yolo11s_fine_tune_v3.pt.validation.json`; do not enable alerts for a model whose validation results or sample predictions are unacceptable. A generic model or unvalidated six-class checkpoint cannot send threat alerts.

The six-class dataset configuration is `The_final-4/data.yaml` (14,018 train, 4,005 validation, and 2,002 test images). Training is CPU-only here and can take many hours. The training script creates a timestamped output directory so existing runs are not overwritten:

```powershell
.\.venv\Scripts\python.exe scripts\train_weapon_cpu.py
```

The optional action checkpoint path is `models/lstm_action_recognition_pro_v2.pth`. Action inference requires a compatible bidirectional LSTM checkpoint and consecutive sequences of 30 valid poses. Each pose frame is 132 `float32` values in MediaPipe landmark order, with `(x, y, z, visibility)` per landmark, estimated from the detected person crop. Missing/invalid poses break the sequence; no prediction is made before a complete sequence is collected. Checkpoints record the sequence length, feature layout, architecture, dropout, and class map, all of which the loader validates along with tensor shapes.

This workspace currently has neither the configured LSTM checkpoint nor `data/processed/lstm_data`, so fighting/shooting recognition is unavailable. Do not use `scripts/train_lstm.py` for the deployed model: it trains a legacy unidirectional architecture and its checkpoint is incompatible. The supported training path requires labeled clip folders named exactly `normal`, `fighting`, and `shooting`:

```powershell
$env:SMARTVISION_ACTION_VIDEO_DIR = "C:\path\to\labeled_action_clips"
\.venv\Scripts\python.exe scripts\create_lstm_data.py
\.venv\Scripts\python.exe scripts\train_lstm_pro.py
```

Feature generation writes only complete consecutive valid-pose windows. Training rejects missing classes, malformed arrays, wrong feature dimensions, and non-finite values. Training was not run because the labeled clips and generated feature data are absent.

The Streamlit server permits one session per process to own the webcam. Dashboard timing reports use measured capture/inference rates and rolling stage mean/p95 values. The default inference size is 416: on the project CPU, the generic COCO model measured approximately 212 ms/frame at 640 and 102 ms/frame at 416 after warm-up. This is a performance result, not an accuracy validation; use the validation dataset with the trained six-class checkpoint before production deployment. CPU thread count is left unchanged. A live frame older than 500 ms is withheld and its age/error is shown until capture recovers.

MediaPipe 1.1 uses the Tasks API rather than `solutions.pose`. Download the official Lite pose model once from the project root:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
curl.exe --fail --location --output models\pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

The downloaded file is ignored by Git; the application reports the full model path and an actionable diagnostic if it is missing.

For a manual Windows launch from the project root, open two PowerShell terminals:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run pipeline\app.py --server.port 8501
```

---

## 🛡️ License & Acknowledgments

Developed as part of the SmartVision real-time computer vision security platform.
