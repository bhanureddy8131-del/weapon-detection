import json
import math
import os
import sys
import time
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import torch

# ============================================================
# SMARTVISION PRO - PROJECT SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    API_BASE_URL,
    CLASS_CONFIDENCE_THRESHOLDS,
    CONFIDENCE_THRESHOLD,
    INPUT_VIDEO_DIR,
    WEBCAM_ID,
    YOLO_IMAGE_SIZE,
)

from pipeline.run_inference import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    get_canonical_class_id,
    load_models,
)

from pipeline.inference_worker import InferenceWorker
from pipeline.latest_frame_capture import LatestFrameCapture
from pipeline.analytics_dashboard import render_analytics_dashboard


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartVision Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = f"{API_BASE_URL}/incidents"
HEALTH_URL = f"{API_BASE_URL}/health"
NOTIFICATION_STATUS_URL = f"{API_BASE_URL}/notifications/status"
TEST_NOTIFICATION_URL = f"{API_BASE_URL}/notifications/test"

VALIDATION_REPORT_PATH = (
    ROOT_DIR / "runs" / "validation" / "latest" / "validation_report.json"
)

DATASET_AUDIT_ROOT = ROOT_DIR / "runs" / "dataset_audit"

IMAGE_SIZE_OPTIONS = [320, 416, 512, 640, 768, 960, 1280]


# ============================================================
# SMARTVISION PRO - DARK PREMIUM THEME
# ============================================================

def apply_custom_design():
    """Apply the SmartVision Pro dark navy, purple and teal theme."""
    st.markdown(
        """
        <style>
        :root {
            --sv-bg: #090914;
            --sv-bg-soft: #101022;
            --sv-sidebar: #17112D;
            --sv-card: #17172D;
            --sv-card-hover: #20203B;
            --sv-purple: #A78BFA;
            --sv-purple-dark: #7C3AED;
            --sv-teal: #2DD4BF;
            --sv-mint: #6EE7B7;
            --sv-text: #F8FAFC;
            --sv-muted: #A1A1B5;
            --sv-border: #30304D;
            --sv-danger: #FB7185;
            --sv-warning: #FBBF24;
            --sv-success: #34D399;
        }

        /* App background and layout */
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: var(--sv-bg) !important;
            color: var(--sv-text) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(9, 9, 20, 0.92) !important;
        }

        [data-testid="stMainBlockContainer"],
        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #17112D 0%,
                #211640 56%,
                #102B35 100%
            ) !important;
            border-right: 1px solid #383052;
        }

        [data-testid="stSidebar"] * {
            color: #F8FAFC;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #C4BBDD !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(255,255,255,0.025);
            border-radius: 9px;
            padding: 7px 10px;
            margin-bottom: 4px;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(167,139,250,0.16);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            color: var(--sv-text) !important;
            font-weight: 700 !important;
        }

        p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
            color: var(--sv-muted);
        }

        [data-testid="stCaptionContainer"] * {
            color: var(--sv-muted) !important;
        }

        /* Page banner */
        .sv-header {
            background: linear-gradient(
                115deg,
                #241443 0%,
                #252052 55%,
                #103C43 100%
            );
            color: white;
            padding: 27px 30px;
            border: 1px solid #49366E;
            border-radius: 18px;
            margin-bottom: 24px;
            box-shadow: 0 12px 34px rgba(0,0,0,0.28);
        }

        .sv-header h1 {
            color: #FFFFFF !important;
            font-size: 30px;
            margin: 0;
        }

        .sv-header p {
            color: #D5CDED !important;
            margin: 8px 0 0 0;
        }

        .sv-section {
            font-size: 19px;
            font-weight: 750;
            color: #F8FAFC !important;
            margin-top: 15px;
            margin-bottom: 12px;
        }

        .sv-caption {
            color: var(--sv-muted) !important;
            font-size: 13px;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: var(--sv-card) !important;
            border: 1px solid var(--sv-border) !important;
            border-radius: 15px !important;
            padding: 17px 18px !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
            transition: transform 0.18s ease, border-color 0.18s ease;
        }

        [data-testid="stMetric"]:hover {
            border-color: var(--sv-purple) !important;
            transform: translateY(-2px);
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] * {
            color: var(--sv-muted) !important;
            font-size: 13px;
            font-weight: 600;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            color: var(--sv-purple) !important;
            font-weight: 800;
        }

        [data-testid="stMetricDelta"] {
            color: var(--sv-mint) !important;
        }

        /* Bordered containers and cards */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--sv-card) !important;
            border: 1px solid var(--sv-border) !important;
            border-radius: 15px !important;
        }

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(110deg, #7C3AED, #6850C9) !important;
            color: #FFFFFF !important;
            border: 1px solid #9272E8 !important;
            border-radius: 10px !important;
            min-height: 40px;
            font-weight: 650;
            transition: all 0.2s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(110deg, #8B5CF6, #14B8A6) !important;
            border-color: var(--sv-teal) !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
        }

        /* Text inputs, number inputs and text areas */
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stDateInput input,
        .stTimeInput input {
            background: #101022 !important;
            color: #FFFFFF !important;
            border: 1px solid var(--sv-border) !important;
            border-radius: 9px !important;
        }

        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stTextArea textarea:focus {
            border-color: var(--sv-purple) !important;
            box-shadow: 0 0 0 1px var(--sv-purple) !important;
        }

        /* Selectbox and multiselect */
        [data-baseweb="select"] > div {
            background: #17172D !important;
            color: #FFFFFF !important;
            border-color: var(--sv-border) !important;
            border-radius: 9px !important;
        }

        [data-baseweb="select"] * {
            color: #FFFFFF !important;
        }

        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [data-baseweb="listbox"] {
            background: #17172D !important;
            color: #FFFFFF !important;
        }

        [data-baseweb="menu"] * {
            color: #FFFFFF !important;
        }

        /* Sliders */
        [data-testid="stSlider"] [data-baseweb="slider"] div {
            color: var(--sv-purple);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: #141427;
            border: 1px solid var(--sv-border);
            border-radius: 12px;
            padding: 6px;
            gap: 7px;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: var(--sv-muted) !important;
            border-radius: 8px;
            padding: 9px 16px;
        }

        .stTabs [aria-selected="true"] {
            background: #39265F !important;
            color: #FFFFFF !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background: var(--sv-card) !important;
            border: 1px solid var(--sv-border) !important;
            border-radius: 12px !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            color: var(--sv-text) !important;
        }

        /* Alerts */
        [data-testid="stAlert"] {
            background: #19192E !important;
            border: 1px solid var(--sv-border) !important;
            border-radius: 11px !important;
            color: var(--sv-text) !important;
        }

        [data-testid="stAlert"] p {
            color: var(--sv-text) !important;
        }

        /* Dataframes and tables */
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: var(--sv-card) !important;
            border: 1px solid var(--sv-border) !important;
            border-radius: 12px !important;
            overflow: hidden;
        }

        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(
                90deg,
                var(--sv-purple),
                var(--sv-teal)
            ) !important;
        }

        /* Checkboxes and radio controls */
        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label {
            color: var(--sv-text) !important;
        }

        /* Dividers */
        hr {
            border-color: var(--sv-border) !important;
        }

        /* Status badges */
        [data-testid="stBadge"] {
            border-radius: 20px;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background: var(--sv-card);
            border: 1px dashed #514078;
            border-radius: 12px;
            padding: 10px;
        }

        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #101022;
        }

        ::-webkit-scrollbar-thumb {
            background: #514078;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--sv-purple);
        }

        /* Hide Streamlit footer/menu */
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 1rem;
            }

            .sv-header {
                padding: 20px;
            }

            .sv-header h1 {
                font-size: 24px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_page_banner(title, subtitle):
    st.markdown(
        f"""
        <div class="sv-header">
            <h1>🛡️ {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BACKEND HEALTH MONITOR
# ============================================================

class APIHealthMonitor:

    def __init__(self, interval_seconds=5.0):
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._status = {
            "healthy": None,
            "database_status": "checking",
            "database_error": None,
            "notification_status": "unknown",
            "notification_error": None,
            "notification_details": None,
            "latency_ms": None,
            "error": None,
            "checked_at": None,
        }

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="smartvision-api-health",
        )

        self._thread.start()

    def _run(self):

        while not self._stop_event.is_set():

            started = time.perf_counter()

            try:
                response = requests.get(
                    HEALTH_URL,
                    timeout=(0.5, 2.0),
                )

                response.raise_for_status()
                data = response.json()

                if not isinstance(data, dict):
                    raise ValueError("Invalid health response.")

                healthy = data.get("status") == "running"

                database_status = data.get(
                    "database_status",
                    "unknown",
                )

                database_error = data.get("database_error")
                notification_status = data.get(
                    "notification_status",
                    "unknown",
                )

                notification_error = data.get(
                    "notification_error"
                )

                error = None

            except (requests.RequestException, ValueError) as exc:

                healthy = False
                database_status = "unavailable"
                database_error = None
                notification_status = "unknown"
                notification_error = None
                error = str(exc)

            notification_details = None

            if healthy:

                try:
                    response = requests.get(
                        NOTIFICATION_STATUS_URL,
                        timeout=(0.5, 2.0),
                    )

                    response.raise_for_status()
                    notification_details = response.json()

                    if not isinstance(notification_details, dict):
                        raise ValueError(
                            "Invalid notification status response."
                        )

                    notification_status = notification_details.get(
                        "transport_status",
                        notification_status,
                    )

                    notification_error = notification_details.get(
                        "transport_error"
                    )

                except (requests.RequestException, ValueError) as exc:

                    notification_status = "unknown"
                    notification_error = str(exc)

            with self._lock:

                self._status = {
                    "healthy": healthy,
                    "database_status": database_status,
                    "database_error": database_error,
                    "notification_status": notification_status,
                    "notification_error": notification_error,
                    "notification_details": notification_details,
                    "latency_ms": (
                        time.perf_counter() - started
                    ) * 1000,
                    "error": error,
                    "checked_at": datetime.now(),
                }

            self._stop_event.wait(self.interval_seconds)

    def snapshot(self):

        with self._lock:
            return dict(self._status)


@st.cache_resource
def get_api_health_monitor():
    return APIHealthMonitor()


@st.cache_resource
def get_webcam_lock():
    return threading.Lock()


# ============================================================
# MODEL INITIALIZATION
# ============================================================

@st.cache_resource
def init_models_and_helpers():

    detector, lstm_model, device, inv_class_map, threat_manager = (
        load_models()
    )

    if detector is None:
        return None

    return {
        "detector": detector,
        "lstm_model": lstm_model,
        "device": device,
        "inv_class_map": inv_class_map,
        "threat_manager": threat_manager,
        "inference_lock": threading.Lock(),
    }


def available_class_ids(detector):

    names = detector.class_names or {}

    if isinstance(names, dict):
        raw_ids = names.keys()
    else:
        raw_ids = range(len(names))

    return {
        get_canonical_class_id(detector, int(class_id))
        for class_id in raw_ids
    }


# ============================================================
# SOURCE AND WORKER MANAGEMENT
# ============================================================

def stop_active_source():

    worker = st.session_state.get("smartvision_worker")
    capture = st.session_state.get("smartvision_capture")

    worker_stopped = worker is None or worker.stop()
    capture_stopped = capture is None or capture.stop()

    if not worker_stopped or not capture_stopped:
        return False

    st.session_state.pop("smartvision_worker", None)
    st.session_state.pop("smartvision_capture", None)
    st.session_state.pop("smartvision_source", None)
    st.session_state.pop("smartvision_live_sequence", None)

    if st.session_state.pop(
        "smartvision_webcam_lock_held",
        False,
    ):
        get_webcam_lock().release()

    return True


def current_image_size():

    value = st.session_state.get(
        "smartvision_image_size",
        YOLO_IMAGE_SIZE,
    )

    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 320

    if value not in IMAGE_SIZE_OPTIONS:
        value = 320

    return value


def worker_settings(detector):

    return {
        "confidence_threshold": st.session_state.get(
            "smartvision_confidence_threshold",
            CONFIDENCE_THRESHOLD,
        ),
        "image_size": current_image_size(),
        "class_thresholds": dict(
            st.session_state.get(
                "smartvision_class_thresholds",
                CLASS_CONFIDENCE_THRESHOLDS,
            )
        ),
        "enabled_classes": set(
            st.session_state.get(
                "smartvision_enabled_classes",
                available_class_ids(detector),
            )
        ),
    }


# ============================================================
# COMMON HEADER
# ============================================================

def render_status_header(runtime):

    health = get_api_health_monitor().snapshot()

    capture = st.session_state.get("smartvision_capture")

    detector = runtime.get("detector") if runtime else None

    columns = st.columns([2, 1, 1, 1, 1, 1])

    columns[0].markdown("### 🛡️ SmartVision Pro")

    camera_running = (
        capture is not None and capture.is_running
    )

    camera_open = (
        camera_running and getattr(capture, "opened", False)
    )

    columns[1].badge(
        "Camera connected" if camera_open else "Camera idle",
        color="green" if camera_open else "gray",
    )

    if detector is None:
        detector_text = "Detector unavailable"
        detector_color = "red"

    elif getattr(detector, "is_custom_model", False):

        if not getattr(detector, "is_validated", False):
            detector_text = "Model validation failed"
            detector_color = "red"

        elif not getattr(
            detector,
            "production_alerts_approved",
            False,
        ):
            detector_text = "Approval missing"
            detector_color = "orange"

        else:
            detector_text = "Model approved"
            detector_color = "green"

    else:
        detector_text = "Generic model"
        detector_color = "orange"

    columns[2].badge(
        detector_text,
        color=detector_color,
    )

    api_online = health.get("healthy") is True

    columns[3].badge(
        "API online" if api_online else "API offline",
        color="green" if api_online else "red",
    )

    db_status = health.get("database_status", "unknown")

    columns[4].badge(
        "Storage online" if db_status == "connected"
        else "Storage offline",
        color="green" if db_status == "connected" else "gray",
    )

    details = health.get("notification_details")

    auto_enabled = (
        details.get("automatic_alerts_enabled")
        if isinstance(details, dict)
        else None
    )

    columns[5].badge(
        "Auto alerts ON" if auto_enabled is True
        else "Auto alerts BLOCKED",
        color="green" if auto_enabled is True else "orange",
    )


# ============================================================
# INCIDENT API HELPERS
# ============================================================

def fetch_incidents(limit=100):

    try:

        response = requests.get(
            API_URL,
            params={"limit": limit},
            timeout=(1.0, 5.0),
        )

        if response.status_code != 200:
            return [], (
                f"Incident API returned HTTP {response.status_code}."
            )

        data = response.json()

        if isinstance(data, dict):

            for key in ("items", "incidents", "data"):

                if key in data:
                    data = data[key]
                    break

            else:
                return [], "Unexpected incident API response."

        if not isinstance(data, list):
            return [], "Unexpected incident API response."

        return data, None

    except requests.RequestException as exc:
        return [], f"Could not connect to FastAPI: {exc}"

    except (ValueError, TypeError) as exc:
        return [], f"Could not read incident response: {exc}"


def fetch_incident_summary():

    try:

        response = requests.get(
            f"{API_URL}/summary",
            timeout=(1.0, 5.0),
        )

        response.raise_for_status()

        summary = response.json()

        if not isinstance(summary, dict):
            raise ValueError("Invalid summary response.")

        if (
            type(summary.get("total")) is not int
            or type(summary.get("today")) is not int
            or not isinstance(summary.get("by_class"), dict)
        ):
            raise ValueError("Malformed incident summary.")

        return summary, None

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as exc:

        return None, str(exc)


def normalize_incidents(alerts):

    records = []
    invalid = 0

    for alert in alerts:

        if not isinstance(alert, dict):
            invalid += 1
            continue

        row = dict(alert)

        try:
            row["confidence"] = (
                float(row["confidence"])
                if row.get("confidence") is not None
                else None
            )
        except (TypeError, ValueError):
            row["confidence"] = None

        created_at = row.get("created_at")

        row["created_at"] = (
            pd.to_datetime(
                str(created_at),
                errors="coerce",
                utc=True,
            )
            if created_at is not None
            else pd.NaT
        )

        row["threat_level"] = str(
            row.get("threat_level") or "Unknown"
        )

        row["label"] = str(
            row.get("label") or "Unknown"
        )

        row["detected_class"] = str(
            row.get("detected_class") or row["label"]
        )

        row["camera_id"] = str(
            row.get("camera_id") or "Unknown"
        )

        row["notification_status"] = str(
            row.get("notification_status") or "unknown"
        )

        row["notification_error"] = str(
            row.get("notification_error") or ""
        )

        records.append(row)

    return pd.DataFrame(records), invalid


def send_test_notification():

    try:

        response = requests.post(
            TEST_NOTIFICATION_URL,
            timeout=(1.0, 8.0),
        )

        if response.status_code >= 400:

            try:
                detail = response.json().get(
                    "detail",
                    "Notification test failed.",
                )
            except (ValueError, AttributeError):
                detail = "Notification test failed."

            return False, str(detail)

        result = response.json()

        if (
            not isinstance(result, dict)
            or result.get("status") != "sent"
        ):
            return False, "Backend did not confirm acceptance."

        return True, result.get(
            "message",
            "Test notification accepted.",
        )

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as exc:

        return False, str(exc)


# ============================================================
# LIVE DETECTION
# ============================================================

@st.fragment(run_every="500ms")
def render_live_inference():

    capture = st.session_state.get("smartvision_capture")
    worker = st.session_state.get("smartvision_worker")

    runtime = init_models_and_helpers()

    if runtime is None:
        st.error("Detector runtime unavailable.")
        return

    if capture is None or worker is None:
        st.info("Choose a source and press Start.")
        return

    detector = runtime["detector"]

    worker.update_settings(worker_settings(detector))

    result = worker.snapshot()

    state = result.get("state_data", {})

    health = get_api_health_monitor().snapshot()

    threat_manager = runtime["threat_manager"]

    delivery = threat_manager.get_delivery_health()

    # Status cards

    columns = st.columns(6)

    columns[0].badge(
        "YOLO ready" if getattr(
            detector,
            "alerts_enabled",
            False,
        ) else "YOLO review required",
        color="green" if getattr(
            detector,
            "alerts_enabled",
            False,
        ) else "orange",
    )

    lstm_ready = runtime["lstm_model"] is not None

    columns[1].badge(
        "LSTM ready" if lstm_ready else "LSTM unavailable",
        color="green" if lstm_ready else "orange",
    )

    pose = worker.pose_estimator

    pose_ready = pose is not None and pose.available

    columns[2].badge(
        "MediaPipe ready" if pose_ready
        else "MediaPipe unavailable",
        color="green" if pose_ready else "orange",
    )

    columns[3].badge(
        "ByteTrack running" if worker.is_running
        else "ByteTrack stopped",
        color="green" if worker.is_running else "gray",
    )

    columns[4].badge(
        "API online" if health.get("healthy") is True
        else "API offline",
        color="green" if health.get("healthy") is True else "red",
    )

    details = health.get("notification_details")

    auto_enabled = (
        details.get("automatic_alerts_enabled")
        if isinstance(details, dict)
        else None
    )

    columns[5].badge(
        "Alerts enabled" if auto_enabled is True
        else "Alerts blocked",
        color="green" if auto_enabled is True else "orange",
    )

    if auto_enabled is False and isinstance(details, dict):

        st.warning(
            "Automatic notifications are blocked: "
            + str(
                details.get("blocked_reason")
                or "Review Phone alerts."
            )
        )

    if not lstm_ready:
        st.info(
            "Action recognition is unavailable until a compatible "
            "LSTM checkpoint is installed."
        )

    if lstm_ready and not pose_ready:

        pose_error = (
            getattr(pose, "diagnostic", None)
            if pose else None
        )

        st.warning(
            "MediaPipe pose estimation is unavailable. "
            + str(pose_error or "")
        )

    # Latest annotated frame

    live_frame = result.get("last_processed_frame")

    captured_at = result.get("last_captured_at")

    live_age_ms = (
        max(
            0.0,
            (time.perf_counter() - captured_at) * 1000,
        )
        if captured_at is not None
        else None
    )

    st.markdown(
        '<div class="sv-section">📹 Live camera feed</div>',
        unsafe_allow_html=True,
    )

    if live_frame is not None:

        st.image(
            live_frame,
            channels="BGR",
            width="stretch",
        )

    elif capture.is_webcam:
        st.warning("Waiting for the first webcam frame.")

    else:
        st.info("Waiting for the first video frame.")

    if capture.error and capture.error != "Video finished.":
        st.error(capture.error)

    elif capture.error == "Video finished.":
        st.info("Video finished.")

    if result.get("error"):
        st.error(result["error"])

    # Detection metrics

    detections = state.get("object_detections", [])
    people = state.get("person_detections", [])

    weapons = [
        item for item in detections
        if item.get("class_id") in (2, 4)
        and item.get("is_project_class", False)
    ]

    st.markdown(
        '<div class="sv-section">📊 Live detection metrics</div>',
        unsafe_allow_html=True,
    )

    metrics = st.columns(5)

    metrics[0].metric(
        "Total detections",
        len(detections) + len(people),
        border=True,
    )

    metrics[1].metric(
        "People",
        state.get("people_detected", 0),
        border=True,
    )

    metrics[2].metric(
        "Weapon detections",
        len(weapons) if getattr(
            detector,
            "weapon_detection_available",
            False,
        ) else "Unavailable",
        border=True,
    )

    metrics[3].metric(
        "Inference FPS",
        f"{result.get('inference_fps', 0.0):.1f}",
        border=True,
    )

    metrics[4].metric(
        "Inference latency",
        f"{result.get('frame_latency_ms', 0.0):.0f} ms",
        border=True,
    )

    if not getattr(
        detector,
        "weapon_detection_available",
        False,
    ):

        st.warning(
            "The active model has not been confirmed to support "
            "the project's trained weapon classes."
        )

    extra = st.columns(4)

    extra[0].metric(
        "Capture FPS",
        f"{capture.capture_fps:.1f}",
        border=True,
    )

    extra[1].metric(
        "Frame age",
        f"{live_age_ms:.0f} ms"
        if live_age_ms is not None
        else "Unavailable",
        border=True,
    )

    extra[2].metric(
        "Processing time",
        f"{result.get('processing_ms', 0.0):.0f} ms",
        border=True,
    )

    extra[3].metric(
        "Dropped frames",
        capture.dropped_frames,
        border=True,
    )

    st.caption(
        f"Capture resolution: {result.get('capture_resolution')} "
        f"· Inference resolution: {result.get('inference_resolution')} "
        f"· YOLO size: {current_image_size()}"
    )

    # Performance timings

    timings = result.get("timing_summary_ms", {})

    if timings:

        st.subheader("Inference performance")

        stages = [
            "yolo",
            "preprocess",
            "mediapipe",
            "lstm",
            "bytetrack",
            "postprocess",
            "alert_enqueue",
            "total_processing",
        ]

        visible = [
            stage for stage in stages
            if stage in timings
        ]

        for start in range(0, len(visible), 4):

            stage_columns = st.columns(
                min(4, len(visible) - start)
            )

            for column, stage in zip(
                stage_columns,
                visible[start:start + 4],
            ):

                summary = timings[stage]

                column.metric(
                    stage.replace("_", " ").title(),
                    f"{summary.get('mean', 0.0):.1f} ms",
                    delta=f"P95 {summary.get('p95', 0.0):.1f} ms",
                )

    # Recent detections

    st.subheader("Recent detections")

    recent = result.get("recent_detections", [])

    if recent:

        rows = [
            {
                "Timestamp": item.get("timestamp"),
                "Class": item.get("class"),
                "Confidence": item.get("confidence"),
                "Track ID": item.get("track_id"),
            }
            for item in recent
            if isinstance(item, dict)
        ]

        st.dataframe(
            pd.DataFrame(rows).tail(20),
            hide_index=True,
            width="stretch",
        )

    else:
        st.info("No detections reported yet.")

    # Action recognition

    st.subheader("Action recognition")

    actions = state.get("action_detections", [])

    if actions:

        for action in actions:

            st.write(
                f"{action.get('class', 'Unknown')} · "
                f"{action.get('confidence', 0.0):.1%} · "
                f"Track ID {action.get('track_id', 'Unknown')}"
            )

    elif not lstm_ready:
        st.warning("LSTM action recognition is unavailable.")

    else:
        st.caption("No non-normal action reported.")

    # Session incident history

    with st.expander("Session incident history"):

        incidents = threat_manager.get_incident_history(
            limit=10
        )

        if not incidents:
            st.info("No incidents in this session.")

        for incident in incidents:

            st.write(
                f"{incident.get('created_at')} · "
                f"{incident.get('event_type')} · "
                f"{incident.get('label')} · "
                f"{incident.get('status')}"
            )

            if incident.get("snapshot"):
                st.image(
                    incident["snapshot"],
                    width=320,
                )

            if incident.get("error"):
                st.caption(incident["error"])

        st.caption(
            f"Alerts waiting to send: {delivery.get('queued', 0)}"
        )


@st.fragment(run_every="1s")
def render_video_progress():

    capture = st.session_state.get("smartvision_capture")
    worker = st.session_state.get("smartvision_worker")

    if capture is None or worker is None:
        return

    frame_count = getattr(capture, "frame_count", 0)

    if frame_count > 0:

        processed = worker.snapshot().get(
            "processed_frames",
            0,
        )

        progress = min(1.0, processed / frame_count)

        st.progress(
            progress,
            text=(
                f"Processed {min(processed, frame_count):,} "
                f"of {frame_count:,} frames"
            ),
        )

    else:
        st.caption("Video processing is running.")


# ============================================================
# LIVE DETECTION CONTROLS
# ============================================================

def run_streamlit_inference_page(video_analysis=False):

    if video_analysis:

        render_page_banner(
            "Video analysis",
            "Analyze recorded videos using the SmartVision inference pipeline.",
        )

        source_key = "smartvision_video_source_choice"
        default_source = "Upload Video"

    else:

        render_page_banner(
            "Live detection",
            "Real-time YOLO detection, tracking and pose analysis.",
        )

        source_key = "smartvision_live_source_choice"
        default_source = "Live Webcam"

    runtime = init_models_and_helpers()

    if runtime is None:
        st.error("YOLO model could not be loaded.")
        return

    detector = runtime["detector"]

    class_ids = available_class_ids(detector)

    if getattr(detector, "alerts_enabled", False):
        st.sidebar.success("Validated detector loaded.")

    else:
        st.sidebar.warning(
            getattr(
                detector,
                "model_diagnostic",
                "Detector validation is required.",
            )
        )

    if st.sidebar.button("Reload model and settings"):

        if not stop_active_source():
            st.error("Stop the active source before reloading.")
            return

        old_runtime = init_models_and_helpers()

        if old_runtime:
            old_runtime["threat_manager"].close()

        init_models_and_helpers.clear()

        st.rerun()

    source_options = [
        "Live Webcam",
        "Select from Input Folder",
        "Upload Video",
    ]

    if source_key not in st.session_state:
        st.session_state[source_key] = default_source

    input_option = st.sidebar.selectbox(
        "Input source",
        source_options,
        key=source_key,
    )

    source_path = None

    if input_option == "Live Webcam":

        source_path = int(
            st.session_state.get(
                "smartvision_camera_index",
                WEBCAM_ID,
            )
        )

        st.info("Live webcam selected. Press Start.")

    elif input_option == "Select from Input Folder":

        if os.path.isdir(INPUT_VIDEO_DIR):

            files = sorted(
                name for name in os.listdir(INPUT_VIDEO_DIR)
                if name.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                )
            )

            if files:

                selected = st.sidebar.selectbox(
                    "Choose video",
                    files,
                )

                source_path = os.path.join(
                    INPUT_VIDEO_DIR,
                    selected,
                )

                st.info(f"Selected video: {selected}")

            else:
                st.warning("No videos found in the input folder.")

        else:
            st.error(f"Input folder not found: {INPUT_VIDEO_DIR}")

    elif input_option == "Upload Video":

        uploaded = st.sidebar.file_uploader(
            "Choose video",
            type=["mp4", "avi", "mov", "mkv"],
        )

        if uploaded is not None:

            signature = (uploaded.name, uploaded.size)

            old_path = st.session_state.get(
                "smartvision_uploaded_temp_path"
            )

            old_signature = st.session_state.get(
                "smartvision_uploaded_signature"
            )

            try:

                if (
                    old_path
                    and old_signature == signature
                    and os.path.isfile(old_path)
                ):

                    source_path = old_path

                else:

                    if old_path and os.path.isfile(old_path):
                        os.remove(old_path)

                    suffix = os.path.splitext(uploaded.name)[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_file:

                        temp_file.write(uploaded.getbuffer())
                        source_path = temp_file.name

                    st.session_state[
                        "smartvision_uploaded_temp_path"
                    ] = source_path

                    st.session_state[
                        "smartvision_uploaded_signature"
                    ] = signature

                st.success("Video is ready.")

            except OSError as exc:
                st.error(str(exc))

    start_col, stop_col = st.sidebar.columns(2)

    if start_col.button(
        "Start",
        type="primary",
        key="start_smartvision",
    ):

        if source_path is None:

            st.error("Select a valid source first.")

        else:

            old_capture = st.session_state.get(
                "smartvision_capture"
            )

            old_worker = st.session_state.get(
                "smartvision_worker"
            )

            already_running = (
                old_capture is not None
                and old_worker is not None
                and old_capture.source == source_path
                and old_capture.is_running
                and old_worker.is_running
            )

            if already_running:

                st.info("This source is already running.")

            elif not stop_active_source():

                st.error("Previous source is still stopping.")

            else:

                lock = get_webcam_lock()
                lock_acquired = False

                if isinstance(source_path, int):

                    lock_acquired = lock.acquire(
                        blocking=False
                    )

                    if not lock_acquired:
                        st.error("Webcam is already in use.")

                if (
                    not isinstance(source_path, int)
                    or lock_acquired
                ):

                    if lock_acquired:

                        st.session_state[
                            "smartvision_webcam_lock_held"
                        ] = True

                    capture = LatestFrameCapture(
                        source_path,
                        width=FRAME_WIDTH,
                        height=FRAME_HEIGHT,
                    )

                    if capture.wait_until_ready():

                        st.session_state[
                            "smartvision_capture"
                        ] = capture

                        if capture.is_webcam:
                            source_name = f"camera_{capture.source}"
                        else:
                            source_name = os.path.basename(
                                str(capture.source)
                            )

                        try:

                            worker = InferenceWorker(
                                capture,
                                runtime,
                                source_name,
                                settings=worker_settings(detector),
                            )

                        except Exception as exc:

                            capture.stop()

                            st.session_state.pop(
                                "smartvision_capture",
                                None,
                            )

                            if lock_acquired:

                                st.session_state.pop(
                                    "smartvision_webcam_lock_held",
                                    None,
                                )

                                lock.release()

                            st.error(
                                f"Could not start inference: {exc}"
                            )

                        else:

                            st.session_state[
                                "smartvision_worker"
                            ] = worker

                            st.session_state[
                                "smartvision_source"
                            ] = source_path

                    else:

                        st.error(
                            capture.error or "Could not open source."
                        )

                        capture.stop()

                        if lock_acquired:

                            st.session_state.pop(
                                "smartvision_webcam_lock_held",
                                None,
                            )

                            lock.release()

    if stop_col.button(
        "Stop",
        key="stop_smartvision",
    ):

        if stop_active_source():

            uploaded_path = st.session_state.pop(
                "smartvision_uploaded_temp_path",
                None,
            )

            if uploaded_path and os.path.isfile(uploaded_path):

                try:
                    os.remove(uploaded_path)
                except OSError as exc:
                    st.warning(str(exc))

            st.session_state.pop(
                "smartvision_uploaded_signature",
                None,
            )

            st.success("Inference stopped.")

        else:
            st.error("Source is still stopping. Try again.")

    capture = st.session_state.get("smartvision_capture")

    if capture is not None:

        if video_analysis and not capture.is_webcam:
            render_video_progress()

        render_live_inference()

    else:
        st.info("No active source. Select a source and press Start.")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def read_json_report(path):

    try:

        with Path(path).open(
            "r",
            encoding="utf-8",
        ) as file:

            report = json.load(file)

    except FileNotFoundError:
        return None, f"Report unavailable: {path}"

    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)

    if not isinstance(report, dict):
        return None, "Invalid report format."

    return report, None


def render_model_performance():

    render_page_banner(
        "Model performance",
        "Review recorded validation metrics and detector acceptance checks.",
    )

    report, error = read_json_report(
        VALIDATION_REPORT_PATH
    )

    if error:

        st.info(
            "No completed validation report is available. "
            "Run the detector evaluator first."
        )

        st.caption(error)
        return

    summary = report.get("validation_summary", {})

    required_metrics = (
        "precision",
        "recall",
        "map50",
        "map50_95",
    )

    metrics_present = (
        isinstance(summary, dict)
        and all(
            isinstance(summary.get(metric), (int, float))
            and not isinstance(summary.get(metric), bool)
            and math.isfinite(float(summary[metric]))
            for metric in required_metrics
        )
    )

    acceptance_checks = report.get("acceptance_checks")

    required_checks = (
        "precision",
        "recall",
        "map50",
        "map50_95",
        "gun_recall",
        "knife_recall",
        "person_recall",
        "representative_tests_complete",
        "no_missed_weapon_detections",
        "weapon_false_positive_limit",
    )

    checks_passed = (
        isinstance(acceptance_checks, dict)
        and all(
            isinstance(acceptance_checks.get(name), dict)
            and acceptance_checks[name].get("passed") is True
            for name in required_checks
        )
    )

    representative = report.get("representative_tests")

    passed = (
        report.get("validation_completed") is True
        and report.get("validation_passed") is True
        and report.get("evaluation_split") == "val"
        and metrics_present
        and checks_passed
        and isinstance(representative, dict)
        and representative.get("completed") is True
    )

    if passed:
        st.success("Recorded validation passed.")
    else:
        st.error(
            "Validation did not pass. Production approval and "
            "automatic alerts must remain disabled."
        )

    st.caption(
        f"Checkpoint: {report.get('checkpoint_path', 'Unavailable')}"
    )

    st.caption(
        f"SHA-256: {report.get('checkpoint_sha256', 'Unavailable')}"
    )

    st.caption(
        f"Completed: {report.get('validation_completed_at', 'Unavailable')}"
    )

    columns = st.columns(4)

    for column, name, label in zip(
        columns,
        required_metrics,
        ("Precision", "Recall", "mAP50", "mAP50-95"),
    ):

        value = summary.get(name)

        column.metric(
            label,
            f"{value:.3f}"
            if isinstance(value, (int, float))
            else "Unavailable",
        )

    class_metrics = report.get("per_class_metrics")

    if isinstance(class_metrics, dict) and class_metrics:

        st.subheader("Per-class metrics")

        rows = [
            {"Class": name, **values}
            for name, values in class_metrics.items()
            if isinstance(values, dict)
        ]

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
            )

    if isinstance(acceptance_checks, dict):

        st.subheader("Acceptance checks")

        rows = []

        for name, value in acceptance_checks.items():

            if not isinstance(value, dict):
                continue

            rows.append({
                "Check": name,
                "Passed": value.get("passed"),
                "Actual": value.get("actual"),
                "Required": value.get(
                    "minimum",
                    value.get(
                        "maximum",
                        value.get("expected"),
                    ),
                ),
            })

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
            )

    if isinstance(representative, dict):

        st.subheader("Representative image tests")

        st.write(
            "Completed:",
            representative.get("completed"),
        )

        st.write(
            "Missed weapon detections:",
            representative.get(
                "missed_weapon_detections",
                "Unavailable",
            ),
        )

        st.write(
            "Weapon false positives:",
            representative.get(
                "weapon_false_positives",
                "Unavailable",
            ),
        )

    confusion_path = report.get("confusion_matrix_image")

    if confusion_path and Path(confusion_path).is_file():

        st.image(
            str(confusion_path),
            caption="Validation confusion matrix",
            width="stretch",
        )


# ============================================================
# DATASET AND TRAINING
# ============================================================

def latest_dataset_audit():

    if not DATASET_AUDIT_ROOT.is_dir():
        return None

    reports = sorted(
        DATASET_AUDIT_ROOT.rglob("dataset_audit.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return reports[0] if reports else None


def render_dataset_training():

    render_page_banner(
        "Dataset and training",
        "Review dataset quality, class distribution and training experiments.",
    )

    audit_path = latest_dataset_audit()

    if audit_path is None:

        st.info("No dataset audit report is available.")

    else:

        audit, error = read_json_report(audit_path)

        if error:
            st.warning(error)

        elif audit:

            if audit.get("audit_passed") is True:
                st.success("Latest dataset audit passed.")
            else:
                st.error("Latest dataset audit did not pass.")

            st.caption(f"Report: {audit_path}")

            split_summary = []

            for split_name, values in audit.get(
                "splits",
                {},
            ).items():

                class_rows = values.get("classes", {})

                valid_annotations = values.get(
                    "valid_annotation_rows",
                    sum(
                        counts.get("instances", 0)
                        for counts in class_rows.values()
                        if isinstance(counts, dict)
                    ),
                )

                split_summary.append({
                    "Split": split_name,
                    "Images": values.get("image_count"),
                    "Readable": values.get("readable_images"),
                    "Labels": values.get("label_files_found"),
                    "Valid annotations": valid_annotations,
                    "Invalid rows": values.get(
                        "invalid_annotation_rows"
                    ),
                })

            if split_summary:

                st.subheader("Dataset split summary")

                st.dataframe(
                    pd.DataFrame(split_summary),
                    hide_index=True,
                    width="stretch",
                )

            class_balance = audit.get("class_balance")

            if isinstance(class_balance, dict):

                st.subheader("Class distribution")

                st.dataframe(
                    pd.DataFrame([
                        {"Class": name, **values}
                        for name, values in class_balance.items()
                    ]),
                    hide_index=True,
                    width="stretch",
                )

            for warning in audit.get("warnings", []):
                st.warning(warning)

    st.subheader("Training experiments")

    runs_root = ROOT_DIR / "runs" / "detect"

    run_directories = sorted(
        (
            path for path in runs_root.glob("smartvision_weapon_*")
            if path.is_dir()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if runs_root.is_dir() else []

    if not run_directories:

        st.info("No SmartVision training experiments recorded.")

    else:

        rows = []

        for path in run_directories:

            checkpoint = path / "weights" / "best.pt"

            rows.append({
                "Experiment": path.name,
                "Completed": (
                    path / "smartvision_training_completion.json"
                ).is_file(),
                "Best checkpoint": (
                    str(checkpoint)
                    if checkpoint.is_file()
                    else "Unavailable"
                ),
            })

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
        )

    st.info(
        "Training is a separate resource-intensive operation. "
        "This dashboard does not automatically start training."
    )


# ============================================================
# ANALYTICS
# ============================================================

def render_analytics():

    render_page_banner(
        "Analytics",
        "Explore recorded incidents and notification delivery outcomes.",
    )

    summary, summary_error = fetch_incident_summary()

    incidents, incident_error = fetch_incidents(
        limit=200
    )

    if summary_error:
        st.warning(summary_error)

    if incident_error:
        st.error(incident_error)
        return

    if summary:

        columns = st.columns(3)

        columns[0].metric(
            "All-time incidents",
            summary["total"],
        )

        columns[1].metric(
            "Incidents today",
            summary["today"],
        )

        columns[2].metric(
            "Recorded classes",
            len(summary["by_class"]),
        )

    frame, invalid = normalize_incidents(incidents)

    if invalid:
        st.warning(f"Ignored {invalid} invalid records.")

    if frame.empty:
        st.info("No stored incidents are available.")
        return

    left, right = st.columns(2)

    with left:

        st.subheader("Incidents by class")

        st.bar_chart(
            frame["label"].value_counts(),
            horizontal=True,
        )

    with right:

        st.subheader("Notification outcomes")

        st.bar_chart(
            frame["notification_status"].value_counts(),
            horizontal=True,
        )

    timed = frame.dropna(
        subset=["created_at"]
    )

    if not timed.empty:

        st.subheader("Incident activity")

        st.line_chart(
            timed.groupby(
                pd.Grouper(
                    key="created_at",
                    freq="h",
                )
            ).size()
        )

    st.caption(
        "This legacy Analytics page displays up to 200 incidents. "
        "Use Analytics Dashboard for the separate historical analytics view."
    )


# ============================================================
# PHONE ALERTS
# ============================================================

def render_phone_alerts():

    render_page_banner(
        "Phone alerts",
        "Monitor notification transport, approval and delivery status.",
    )

    health = get_api_health_monitor().snapshot()

    details = health.get("notification_details")

    if not isinstance(details, dict):

        st.warning(
            "Notification status is unavailable. Check the FastAPI backend."
        )

        return

    values = [
        ("Transport", details.get("transport_status", "unknown")),
        (
            "Notification configuration",
            "Enabled"
            if details.get("notification_configuration_enabled")
            else "Disabled",
        ),
        (
            "Custom model loaded",
            "Yes" if details.get("custom_model_loaded") else "No",
        ),
        (
            "Model validation",
            "Passed" if details.get("model_validation_passed") else "Failed",
        ),
        (
            "Production approval",
            "Granted"
            if details.get("production_alerts_approved")
            else "Missing",
        ),
        (
            "Automatic alerts",
            "Enabled"
            if details.get("automatic_alerts_enabled")
            else "Blocked",
        ),
    ]

    st.dataframe(
        pd.DataFrame([
            {"Gate": name, "Status": value}
            for name, value in values
        ]),
        hide_index=True,
        width="stretch",
    )

    if details.get("blocked_reason"):

        st.error(
            "Automatic alerts are blocked: "
            + str(details["blocked_reason"])
        )

    failures = details.get("validation_failures", [])

    if failures:

        st.subheader("Validation blockers")

        for failure in failures:
            st.warning(str(failure))

    if details.get("transport_status") == "ready":

        st.subheader("Manual notification test")

        st.caption(
            "This sends a test request. Provider acceptance does not "
            "guarantee that the phone received it."
        )

        if st.button(
            "Send test notification",
            type="primary",
        ):

            sent, message = send_test_notification()

            if sent:
                st.success(message)
            else:
                st.error(message)

    st.subheader("Recent delivery attempts")

    incidents, error = fetch_incidents(limit=20)

    if error:

        st.warning(error)

    elif incidents:

        rows = [
            {
                "Created": item.get("created_at"),
                "Class": item.get("detected_class") or item.get("label"),
                "Status": item.get("notification_status", "unknown"),
                "Error": item.get("notification_error"),
            }
            for item in incidents
            if isinstance(item, dict)
        ]

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
        )

    else:

        st.info("No notification attempts are recorded.")


# ============================================================
# OVERVIEW
# ============================================================

@st.fragment(run_every="10s")
def render_overview():

    render_page_banner(
        "Overview",
        "Real-time SmartVision Pro monitoring and incident summary.",
    )

    summary, summary_error = fetch_incident_summary()

    capture = st.session_state.get("smartvision_capture")
    worker = st.session_state.get("smartvision_worker")

    result = worker.snapshot() if worker is not None else {}

    state = result.get("state_data", {})

    weapon_items = state.get("weapon_detections", [])

    columns = st.columns(7)

    columns[0].metric(
        "Active cameras",
        int(
            capture is not None
            and capture.is_webcam
            and capture.is_running
        ),
        border=True,
    )

    columns[1].metric(
        "People detected",
        state.get("people_detected", 0)
        if worker is not None else "Unavailable",
        border=True,
    )

    columns[2].metric(
        "Weapon detections",
        len(weapon_items)
        if worker is not None else "Unavailable",
        border=True,
    )

    columns[3].metric(
        "Total incidents",
        summary["total"] if summary else "Unavailable",
        border=True,
    )

    columns[4].metric(
        "Today's incidents",
        summary["today"] if summary else "Unavailable",
        border=True,
    )

    columns[5].metric(
        "Inference FPS",
        f"{result['inference_fps']:.1f}"
        if worker is not None else "Unavailable",
        border=True,
    )

    timing = result.get(
        "timing_summary_ms",
        {},
    ).get("yolo", {})

    columns[6].metric(
        "YOLO latency",
        f"{timing.get('mean', 0.0):.1f} ms"
        if isinstance(timing, dict) and timing
        else "Unavailable",
        border=True,
    )

    if summary_error:
        st.warning(summary_error)

    incidents, error = fetch_incidents(limit=200)

    if error:
        st.warning(error)
        return

    valid = [
        item for item in incidents
        if isinstance(item, dict)
    ]

    if not valid:

        st.info("No stored incidents available yet.")

    else:

        frame = pd.DataFrame(valid)

        left, right = st.columns(2)

        with left:

            st.subheader("Incidents by severity")

            if "threat_level" in frame:

                st.bar_chart(
                    frame["threat_level"]
                    .fillna("Unknown")
                    .value_counts(),
                    horizontal=True,
                )

        with right:

            st.subheader("Incidents by class")

            if "label" in frame:

                st.bar_chart(
                    frame["label"]
                    .fillna("Unknown")
                    .value_counts(),
                    horizontal=True,
                )

        if "created_at" in frame:

            frame["created_at"] = pd.to_datetime(
                frame["created_at"],
                errors="coerce",
                utc=True,
            )

            timed = frame.dropna(
                subset=["created_at"]
            )

            if not timed.empty:

                st.subheader("Incident activity")

                st.line_chart(
                    timed.groupby(
                        pd.Grouper(
                            key="created_at",
                            freq="h",
                        )
                    ).size()
                )


# ============================================================
# INCIDENT HISTORY
# ============================================================

@st.fragment(run_every="30s")
def render_alert_history():

    render_page_banner(
        "Incident history",
        "Search, filter and export stored detection incidents.",
    )

    summary, summary_error = fetch_incident_summary()

    if summary_error:
        st.warning(summary_error)

    if summary:

        columns = st.columns(3)

        columns[0].metric(
            "Total incidents",
            summary["total"],
        )

        columns[1].metric(
            "Today",
            summary["today"],
        )

        columns[2].metric(
            "Classes recorded",
            len(summary.get("by_class", {})),
        )

    alerts, error = fetch_incidents(limit=200)

    if error:
        st.error(error)
        return

    if not alerts:
        st.info("No incidents found.")
        return

    df, invalid_count = normalize_incidents(alerts)

    if invalid_count:
        st.warning(f"Ignored {invalid_count} invalid records.")

    if df.empty:
        st.info("No valid incident records.")
        return

    search = st.text_input(
        "Search incidents",
        placeholder="Search class, camera or severity...",
    )

    severity_options = sorted(
        df["threat_level"].dropna().unique().tolist()
    )

    class_options = sorted(
        df["label"].dropna().unique().tolist()
    )

    col1, col2 = st.columns(2)

    with col1:

        selected_severity = st.multiselect(
            "Severity",
            severity_options,
            default=severity_options,
        )

    with col2:

        selected_classes = st.multiselect(
            "Class",
            class_options,
            default=class_options,
        )

    filtered = df[
        df["threat_level"].isin(selected_severity)
        & df["label"].isin(selected_classes)
    ].copy()

    if search:

        searchable = filtered[
            ["threat_level", "label", "camera_id"]
        ].fillna("").astype(str).agg(" ".join, axis=1)

        filtered = filtered[
            searchable.str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    columns = [
        column for column in (
            "created_at",
            "threat_level",
            "detected_class",
            "confidence",
            "tracking_id",
            "notification_status",
            "notification_error",
            "camera_id",
            "id",
        )
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[columns],
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download incidents CSV",
        data=filtered[columns].to_csv(
            index=False
        ).encode("utf-8"),
        file_name="smartvision-alerts.csv",
        mime="text/csv",
    )

    st.subheader("Incident details")

    for _, incident in filtered.iterrows():

        title = (
            f"{incident.get('created_at')} · "
            f"{incident.get('threat_level')} · "
            f"{incident.get('label')}"
        )

        with st.expander(title):

            left, right = st.columns([1, 2])

            with left:

                image_path = incident.get("image_path")

                if isinstance(image_path, str) and image_path:

                    candidate = os.path.normpath(
                        image_path
                        if os.path.isabs(image_path)
                        else os.path.join(
                            str(ROOT_DIR),
                            image_path,
                        )
                    )

                    try:

                        safe_path = (
                            os.path.commonpath([
                                os.path.abspath(ROOT_DIR),
                                os.path.abspath(candidate),
                            ])
                            == os.path.abspath(ROOT_DIR)
                        )

                    except ValueError:
                        safe_path = False

                    if safe_path and os.path.isfile(candidate):

                        st.image(
                            candidate,
                            caption="Incident snapshot",
                            width="stretch",
                        )

                    else:
                        st.caption("Snapshot unavailable.")

                else:
                    st.caption("No snapshot attached.")

            with right:

                st.write(
                    "Incident ID:",
                    incident.get("id", "Unknown"),
                )

                st.write(
                    "Camera:",
                    incident.get("camera_id", "Unknown"),
                )

                st.write(
                    "Detected class:",
                    incident.get("detected_class", "Unknown"),
                )

                st.write(
                    "Tracking ID:",
                    incident.get("tracking_id", "Unknown"),
                )

                confidence = incident.get("confidence")

                st.write(
                    "Confidence:",
                    f"{confidence:.1%}"
                    if isinstance(confidence, (float, int))
                    else "Unavailable",
                )

                st.write(
                    "Notification:",
                    incident.get("notification_status", "unknown"),
                )

                if incident.get("notification_error"):
                    st.warning(incident["notification_error"])


# ============================================================
# CAMERA MANAGEMENT
# ============================================================

def render_camera_management():

    render_page_banner(
        "Camera management",
        "Configure the camera device used by SmartVision Pro.",
    )

    st.number_input(
        "Webcam device index",
        min_value=0,
        max_value=16,
        value=int(
            st.session_state.get(
                "smartvision_camera_index",
                WEBCAM_ID,
            )
        ),
        step=1,
        key="smartvision_camera_index",
    )

    capture = st.session_state.get(
        "smartvision_capture"
    )

    with st.container(border=True):

        st.subheader("Camera status")

        if capture is not None and capture.is_running:

            st.success("Camera capture is running.")

            st.metric(
                "Capture FPS",
                f"{capture.capture_fps:.1f}",
            )

            st.write(
                "Resolution:",
                getattr(
                    capture,
                    "resolution",
                    "See live metrics",
                ),
            )

        else:

            st.info("No camera is currently streaming.")

    st.caption(
        "Select Live Webcam under Live detection and press Start."
    )


# ============================================================
# DETECTION SETTINGS
# ============================================================

def render_detection_settings(runtime):

    render_page_banner(
        "Detection settings",
        "Configure YOLO input size, detection classes and confidence thresholds.",
    )

    detector = runtime["detector"]

    class_ids = available_class_ids(detector)

    widget_key = "smartvision_yolo_input_size"

    if widget_key not in st.session_state:

        st.session_state[widget_key] = current_image_size()

    elif st.session_state[widget_key] not in IMAGE_SIZE_OPTIONS:

        st.session_state[widget_key] = 320

    st.subheader("Inference resolution")

    st.select_slider(
        "YOLO input size",
        options=IMAGE_SIZE_OPTIONS,
        key=widget_key,
        help=(
            "320 is a recommended starting point for CPU inference. "
            "Larger values may increase processing time."
        ),
    )

    st.session_state["smartvision_image_size"] = int(
        st.session_state[widget_key]
    )

    st.info(
        f"Selected resolution: "
        f"{st.session_state['smartvision_image_size']} × "
        f"{st.session_state['smartvision_image_size']}"
    )

    st.divider()

    st.subheader("Detection classes")

    options = [
        (5, "Person"),
        (2, "Firearm"),
        (4, "Knife or blade"),
    ]

    options = [
        item for item in options
        if item[0] in class_ids
    ]

    enabled = set(
        st.session_state.get(
            "smartvision_enabled_classes",
            class_ids,
        )
    )

    for class_id, label in options:

        value = st.checkbox(
            f"Enable {label}",
            value=class_id in enabled,
            key=f"enable_class_{class_id}",
        )

        if value:
            enabled.add(class_id)
        else:
            enabled.discard(class_id)

    st.session_state["smartvision_enabled_classes"] = enabled

    st.divider()

    st.subheader("Confidence thresholds")

    global_threshold = float(
        st.session_state.get(
            "smartvision_confidence_threshold",
            CONFIDENCE_THRESHOLD,
        )
    )

    st.session_state[
        "smartvision_confidence_threshold"
    ] = st.slider(
        "Global minimum confidence",
        min_value=0.05,
        max_value=0.99,
        value=min(
            0.99,
            max(0.05, global_threshold),
        ),
        step=0.05,
        key="smartvision_global_confidence",
    )

    thresholds = dict(
        st.session_state.get(
            "smartvision_class_thresholds",
            CLASS_CONFIDENCE_THRESHOLDS,
        )
    )

    for class_id, label in options:

        threshold = float(
            thresholds.get(
                class_id,
                CONFIDENCE_THRESHOLD,
            )
        )

        thresholds[class_id] = st.slider(
            f"{label} confidence",
            min_value=0.05,
            max_value=0.99,
            value=min(
                0.99,
                max(0.05, threshold),
            ),
            step=0.05,
            key=f"class_threshold_{class_id}",
        )

    st.session_state["smartvision_class_thresholds"] = thresholds

    st.divider()

    st.subheader("Incident cooldown")

    manager = runtime["threat_manager"]

    manager.log_cooldown = st.slider(
        "Cooldown per source and event (seconds)",
        min_value=1,
        max_value=300,
        value=int(manager.log_cooldown),
    )

    st.info(
        "Changing input resolution can affect both speed and "
        "small-object detection accuracy."
    )


# ============================================================
# SYSTEM HEALTH
# ============================================================

def render_system_health(runtime):

    render_page_banner(
        "System health",
        "Monitor model status, backend connectivity and notification readiness.",
    )

    st.caption(
        f"Python {sys.version.split()[0]} · "
        f"PyTorch {torch.__version__}"
    )

    if torch.cuda.is_available():

        st.info(
            f"CUDA available: {torch.cuda.device_count()} device(s)."
        )

    else:

        st.info("CUDA unavailable. PyTorch inference uses CPU.")

    if runtime is None or runtime.get("detector") is None:

        st.error("Detector unavailable.")
        return

    detector = runtime["detector"]

    health = get_api_health_monitor().snapshot()

    columns = st.columns(3)

    columns[0].metric(
        "Device",
        str(runtime["device"]),
    )

    columns[1].metric(
        "YOLO input size",
        current_image_size(),
    )

    columns[2].metric(
        "API latency",
        f"{health['latency_ms']:.0f} ms"
        if health.get("latency_ms") is not None
        else "Unavailable",
    )

    st.subheader("YOLO detector")

    if getattr(detector, "is_custom_model", False):

        if getattr(detector, "is_validated", False):
            st.success("Custom detector validation record found.")

        else:

            st.warning(
                getattr(
                    detector,
                    "model_diagnostic",
                    "Custom model has not been validated.",
                )
            )

    else:

        st.warning(
            "A generic model is active. Do not treat its results "
            "as validated weapon detection."
        )

    st.write(
        "Production alerts enabled:",
        bool(getattr(detector, "alerts_enabled", False)),
    )

    st.write(
        "Weapon classes available:",
        bool(
            getattr(
                detector,
                "weapon_detection_available",
                False,
            )
        ),
    )

    names = detector.class_names or {}

    class_names = (
        list(names.values())
        if isinstance(names, dict)
        else list(names)
    )

    st.write(
        "Model classes:",
        ", ".join(map(str, class_names)),
    )

    st.subheader("Action recognition")

    st.write(
        "LSTM:",
        "Loaded"
        if runtime["lstm_model"] is not None
        else "Unavailable",
    )

    worker = st.session_state.get(
        "smartvision_worker"
    )

    pose = (
        worker.pose_estimator
        if worker is not None
        else None
    )

    if runtime["lstm_model"] is None:

        st.info("Compatible LSTM checkpoint is unavailable.")

    elif pose is not None and pose.available:

        st.success("MediaPipe pose estimation is available.")

    else:

        st.warning(
            getattr(
                pose,
                "diagnostic",
                "MediaPipe is not initialized.",
            )
        )

    st.write(
        "ByteTrack:",
        "Running"
        if worker is not None and worker.is_running
        else "Not running",
    )

    st.subheader("Backend and database")

    st.write(
        "Backend:",
        "Online"
        if health["healthy"] is True
        else "Unavailable",
    )

    st.write(
        "Database:",
        health.get("database_status", "unknown"),
    )

    st.write(
        "Notification transport:",
        health.get("notification_status", "unknown"),
    )

    details = health.get("notification_details")

    if isinstance(details, dict):

        st.subheader("Production notification gates")

        gates = [
            (
                "Notification configuration",
                details.get("notification_configuration_enabled"),
            ),
            (
                "Custom detector loaded",
                details.get("custom_model_loaded"),
            ),
            (
                "Model validation passed",
                details.get("model_validation_passed"),
            ),
            (
                "Production approval",
                details.get("production_alerts_approved"),
            ),
            (
                "Automatic alerts",
                details.get("automatic_alerts_enabled"),
            ),
        ]

        st.dataframe(
            pd.DataFrame([
                {
                    "Gate": name,
                    "Status": "Yes" if value else "No",
                }
                for name, value in gates
            ]),
            hide_index=True,
            width="stretch",
        )

        if details.get("blocked_reason"):
            st.warning(details["blocked_reason"])

    if health.get("error"):
        st.warning(health["error"])

    if health.get("database_error"):
        st.warning(health["database_error"])

    if health.get("notification_error"):
        st.warning(health["notification_error"])

    if health.get("notification_status") == "ready":

        if st.button("Send safe test notification"):

            sent, message = send_test_notification()

            if sent:
                st.success(message)
            else:
                st.error(message)

    delivery = runtime["threat_manager"].get_delivery_health()

    st.write(
        "Pending alert queue:",
        delivery.get("queued", 0),
    )

    if not delivery.get("enabled"):

        st.warning(
            "Alert delivery is disabled: "
            + str(delivery.get("disabled_reason"))
        )


# ============================================================
# MAIN NAVIGATION
# ============================================================

def main():

    apply_custom_design()

    runtime = init_models_and_helpers()

    render_status_header(runtime)

    st.sidebar.markdown(
        """
        <div style="padding: 12px 0 18px 0;">
            <h1 style="color:#A78BFA; margin:0;">
                🛡️ SmartVision
            </h1>
            <p style="font-size:12px; color:#C4BBDD;">
                PRO SECURITY MONITORING
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    page = st.sidebar.radio(
        "MAIN MENU",
        [
            "Overview",
            "Live detection",
            "Video analysis",
            "Incident history",
            "Analytics Dashboard",
            "Analytics",
            "Model performance",
            "Dataset and training",
            "Phone alerts",
            "System health",
            "Settings",
        ],
        key="smartvision_navigation",
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "SmartVision Pro · AI-powered monitoring"
    )

    if page == "Overview":

        render_overview()

    elif page == "Live detection":

        run_streamlit_inference_page()

    elif page == "Video analysis":

        run_streamlit_inference_page(
            video_analysis=True
        )

    elif page == "Incident history":

        render_alert_history()

    elif page == "Analytics Dashboard":

        render_analytics_dashboard()

    elif page == "Analytics":

        render_analytics()

    elif page == "Model performance":

        render_model_performance()

    elif page == "Dataset and training":

        render_dataset_training()

    elif page == "Phone alerts":

        render_phone_alerts()

    elif page == "System health":

        render_system_health(runtime)

    elif page == "Settings":

        render_camera_management()

        if runtime is None:

            st.error("Detector could not be loaded.")

        else:

            render_detection_settings(runtime)


if __name__ == "__main__":
    main()