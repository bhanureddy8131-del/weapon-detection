# SmartVision Pro - Real-Time Threat Detection

SmartVision Pro is an advanced real-time computer vision and machine learning platform designed to detect potential security threats such as weapons (guns, firearms, knives, blades), violent activities (fighting, shooting), and context-aware anomalies (ATM activity, unattended bags) using a multi-stage detection pipeline.

---

## 🚀 Key Features

- **Multi-Stage Detection Pipeline**: Integrates YOLO object detection with LSTM action recognition and MediaPipe pose estimation.
- **Weapon & Tool Recognition**: Detects handguns, pistols, rifles, knives, blades, and suspicious objects in real-time.
- **Action Recognition**: Uses skeleton pose dynamics over 30-frame temporal sequences to classify actions.
- **Alert System**: Sends real-time security alerts via Telegram and Email with attached snapshot captures.
- **Interactive Streamlit Dashboard**: Offers live video processing, incident history log, weapon/action toggle options, and custom video uploads.
- **FastAPI Alert Server**: SQLite-backed backend serving alert data, static snapshot files, and REST API endpoints.

---

## ⚡ Quick Start

### 1. Prerequisites
- **Python 3.11**
- **Virtual Environment**: Pre-configured in `env/` or created locally.

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
python -m venv env
# On Windows:
.\env\Scripts\activate
# On Linux/macOS:
source env/bin/activate

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
├── env/                    # Virtual Python environment
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
- `DRAW_PERSON_BOXES`: Enable bounding box overlays for detected persons.
- `DRAW_LABELS`: Enable text labels for detections.
- `SERVER_URL`: Backend endpoint for posting incidents (`http://localhost:8000/api/v1/incidents`).

---

## 🛡️ License & Acknowledgments

Developed as part of the SmartVision real-time computer vision security platform.

