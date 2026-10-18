# SmartVision Pro - Real-Time Threat Detection

SmartVision Pro is a computer vision monitoring platform. Detection and action-recognition capabilities depend on the weight files actually installed; the dashboard reports when it is using a general-purpose fallback model.

---

## 🚀 Key Features

- **Multi-Stage Detection Pipeline**: Integrates YOLO, ByteTrack, optional LSTM action recognition, and MediaPipe Pose estimation.
- **Object Recognition**: Classes are determined by the loaded YOLO weights. Firearm detection requires the project's trained six-class model; the general COCO fallback does not recognize firearms.
- **Action Recognition**: Fighting and shooting recognition require a compatible LSTM checkpoint and pose sequence data.
- **Alert System**: Stores incidents in SQLite and can send authorized phone alerts through opt-in ntfy delivery.
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

## Diagnostics and Local Validation

Run the read-only diagnostic from the project root to report Python and model-library versions, compute device, configured checkpoint status, actual detector classes, dataset class-map agreement, and timed inference on a real test image:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose.py --runs 10
```

To test a specific image or inference configuration:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose.py --image "C:\path\to\sample.jpg" --runs 10 --image-size 416 --confidence 0.10
```

The diagnostic does not write model files or incidents. Its timing and detections are runtime observations, not a validation score. If the custom file is absent or unreadable, the app reports and uses the generic `yolo11s.pt` fallback when available. That fallback does not recognize the project's `gun` or `atm` classes; generic COCO `knife` is not treated as a project weapon. Supply a compatible six-class checkpoint at `models/yolo11s_fine_tune_v3.pt` and validate it against `The_final-4/data.yaml` before relying on those classes.

### Webcam checklist

1. Start FastAPI and Streamlit using the commands below; open `http://localhost:8501` and select **Model and system health**. Confirm the actual detector, API, LSTM, and alert-delivery states.
2. Open **Live detection**, select **Live Webcam**, then select **Start**. Confirm source resolution, inference resolution, capture/inference FPS, latest frame age, and processing latency are updating.
3. Compare the fresh live image with **Latest annotated inference**. Boxes and labels are drawn on the original-resolution frame; detection rows and KPIs come from inference output. A generic detector may show COCO classes, but not project gun/ATM detections.
4. Select **Stop**. Confirm camera access is released before starting another source. To check recovery, temporarily disconnect/re-enable the camera or use a camera that can be safely toggled; the app should show a recovery/error state and resume on new frames.

### Uploaded-video checklist

1. In **Live detection**, choose **Upload Video**, select a readable MP4/AVI/MOV/MKV, and select **Start**.
2. Confirm the annotated inference image advances, detection table and metrics update, and the UI reports decode/inference errors instead of presenting stale frames as current.
3. Select **Stop** and confirm the temporary upload is removed. Repeat with a short clip before testing long videos.

Do not verify production alert delivery by posting synthetic incidents to a live database. Alerts remain disabled until a validated and approved custom detector, verified endpoint, and explicit operator opt-in are all present.

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
| `GET` | `/api/v1/health` | Backend, SQLite, and ntfy configuration health |
| `GET` | `/api/v1/incidents` | List recent incident alerts |
| `GET` | `/api/v1/incidents/summary` | Total incidents and incidents created today |
| `POST` | `/api/v1/incidents` | Persist a validated threat incident and its delivery status |
| `POST` | `/api/v1/notifications/test` | Send an explicit, non-incident ntfy test message |
| `GET` | `/static/alerts/{filename}` | Retrieve snapshot images attached to alert incidents |

---

## 🔧 Configuration Options (`src/config.py`)

- `CONFIDENCE_THRESHOLD`: Object detection confidence threshold (default: `0.10`).
- `ACTION_CONFIDENCE_THRESHOLD`: Minimum confidence for action recognition (default: `0.85`).
- `SMARTVISION_ACTION_CONFIDENCE_THRESHOLD`: Environment override for the action confidence threshold (0 to 1).
- `DRAW_PERSON_BOXES`: Enable bounding box overlays for detected persons.
- `DRAW_LABELS`: Enable text labels for detections.
- `API_BASE_URL`: Incident API base (`SMARTVISION_API_BASE_URL`, default `http://127.0.0.1:8000/api/v1`).
- `SERVER_URL`: Configured POST endpoint (`${API_BASE_URL}/incidents`). FastAPI entry point is `server.main:app`; the project starts it with `python -m uvicorn server.main:app --host 127.0.0.1 --port 8000`.
- `SMARTVISION_ALERTS_ENABLED`: Explicit phone/legacy notification opt-in; defaults to `false`. Incident history is still stored while this is off.
- `SMARTVISION_NTFY_TOPIC`: Private ntfy topic, read from the environment; it is never logged or included in error messages.
- `SMARTVISION_DATABASE_PATH`: Optional SQLite file path. The default is the project-root `data/incidents.db`, shared by FastAPI regardless of its launch directory.
- `SMARTVISION_ALERT_ENDPOINT_VERIFIED`: Separate operator gate; defaults to `false` and must only be enabled after checking the configured endpoint and its OpenAPI multipart contract.
- `SMARTVISION_ALERT_RETRY_ATTEMPTS` and `SMARTVISION_ALERT_RETRY_BACKOFF_SECONDS`: Bounded retry controls (defaults: 3 attempts, 0.5 seconds exponential base; capped at 5 attempts and 2 seconds base).
- `SMARTVISION_YOLO_MODEL_PATH`: Optional path to the trained YOLO checkpoint; defaults to `models/yolo11s_fine_tune_v3.pt`. If it is unavailable, the app reports that weapon detection is unavailable and may use `yolo11s.pt` for general object inference.
- `YOLO_IMAGE_SIZE`: Inference size (`SMARTVISION_YOLO_IMAGE_SIZE`, default `416`; the dashboard allows changing it).
- `YOLO_IOU_THRESHOLD`: YOLO NMS IoU threshold (`SMARTVISION_YOLO_IOU`, default `0.70`).
- `SMARTVISION_CONFIDENCE_THRESHOLD`, `SMARTVISION_PERSON_CONFIDENCE`, `SMARTVISION_GUN_CONFIDENCE`, and `SMARTVISION_KNIFE_CONFIDENCE`: Detection thresholds. The UI can raise, but not lower, the configured global threshold.

## Model files and supported classes

The custom detector path is `models/yolo11s_fine_tune_v3.pt` (or `SMARTVISION_YOLO_MODEL_PATH`). The application reports the actual loaded class IDs and names, checks its six class names, matching weights hash, precision/recall/mAP metrics, and representative validation-image coverage. The training script saves predictions for one validation image per class to a unique run folder. The COCO fallback's generic `knife` prediction is shown as detector-only and is never promoted to the project's validated `knife` threat class. A generic fallback does not detect guns, pistols, or rifles.

When the custom file is absent, the app explicitly falls back to the generic 80-class `yolo11s.pt` COCO model; that model has no `gun` or `atm` class. It can show its `person`, `backpack`, `handbag`, and generic `knife` predictions, but the generic knife is not treated as a project weapon.

Threat incidents are sent to FastAPI on a bounded background queue and saved to the root-relative SQLite database independently of notification opt-in. The API validates threat levels against the project's rule outputs, persists detected class, confidence, track ID, event details, and notification status, and supports idempotent retries. The dashboard reads this same database through `GET /api/v1/incidents`; history survives restarts.

Phone alerts use ntfy. Delivery requires all configured safeguards: approved/validated custom model weights, `SMARTVISION_ALERT_ENDPOINT_VERIFIED=true`, `SMARTVISION_ALERTS_ENABLED=true`, and a non-empty `SMARTVISION_NTFY_TOPIC`. The server stores the incident before attempting ntfy delivery. Notification results are recorded as `pending`, `sent`, `failed`, or `disabled`; only an HTTP 2xx response is treated as sent. Connection and HTTP errors do not remove the incident. The ntfy request has a bounded timeout and runs after API persistence, outside video-frame inference.

To configure it from PowerShell, set the environment before launching both services (replace the topic placeholder locally; do not commit it):

```powershell
$env:SMARTVISION_ALERTS_ENABLED = "false"
$env:SMARTVISION_NTFY_TOPIC = ""
$env:SMARTVISION_DATABASE_PATH = "C:\weapon detection\wepondetection\data\incidents.db"

# After installing approved weights and verifying the API, explicitly opt in:
$env:SMARTVISION_ALERTS_ENABLED = "true"
$env:SMARTVISION_ALERT_ENDPOINT_VERIFIED = "true"
$env:SMARTVISION_NTFY_TOPIC = "<your-private-topic>"
```

The **Send safe test notification** action on Model and system health sends a test message without creating a fake detection incident. Do not use a synthetic incident to verify production storage.

The inference client deduplicates a source/track/event/class during its cooldown and attaches a stable idempotency key to each event window. FastAPI returns the existing record for retries without creating another row or scheduling another notification. A generic model or unvalidated six-class checkpoint can still use the dashboard, but cannot authorize phone alerts.

The six-class dataset configuration is `The_final-4/data.yaml` (14,018 train, 4,005 validation, and 2,002 test images). Training is CPU-only here and can take many hours. The training script creates a timestamped output directory and refuses to overwrite existing application weights or validation metadata:

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

Feature generation writes only complete consecutive valid-pose windows. Training rejects missing classes, malformed arrays, wrong feature dimensions, and non-finite values. The professional LSTM training script refuses to overwrite an existing configured checkpoint. Training was not run because the labeled clips and generated feature data are absent.

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
