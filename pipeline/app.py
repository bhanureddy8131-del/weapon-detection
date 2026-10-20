
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    API_BASE_URL,
    CLASS_CONFIDENCE_THRESHOLDS,
    CONFIDENCE_THRESHOLD,
    INPUT_VIDEO_DIR,
    WEBCAM_ID,
    YOLO_IMAGE_SIZE,
)

from pipeline.run_inference import (
    get_canonical_class_id,
    load_models,
)
from pipeline.inference_worker import InferenceWorker
from pipeline.latest_frame_capture import LatestFrameCapture

API_URL = f"{API_BASE_URL}/incidents"
HEALTH_URL = f"{API_BASE_URL}/health"
NOTIFICATION_STATUS_URL = f"{API_BASE_URL}/notifications/status"
TEST_NOTIFICATION_URL = f"{API_BASE_URL}/notifications/test"
VALIDATION_REPORT_PATH = (
    Path(ROOT_DIR) / "runs" / "validation" / "latest" / "validation_report.json"
)
DATASET_AUDIT_ROOT = Path(ROOT_DIR) / "runs" / "dataset_audit"

IMAGE_SIZE_OPTIONS = [320, 416, 512, 640, 768, 960, 1280]

st.set_page_config(
    page_title="SmartVision Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BACKEND HEALTH MONITOR
# ============================================================

class APIHealthMonitor:
    """Check backend health in a background thread."""

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
            name="smartvision-api-health",
            daemon=True,
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
                    raise ValueError("Invalid backend health response.")

                status = data.get("status") == "running"
                database_status = data.get(
                    "database_status", "unknown"
                )
                database_error = data.get("database_error")
                notification_status = data.get(
                    "notification_status", "unknown"
                )
                notification_error = data.get("notification_error")
                error = None

            except (requests.RequestException, ValueError) as exc:
                status = False
                database_status = "unavailable"
                database_error = None
                notification_status = "unknown"
                notification_error = None
                error = str(exc)

            notification_details = None
            if status:
                try:
                    notification_response = requests.get(
                        NOTIFICATION_STATUS_URL,
                        timeout=(0.5, 2.0),
                    )
                    notification_response.raise_for_status()
                    notification_details = notification_response.json()
                    if not isinstance(notification_details, dict):
                        raise ValueError(
                            "Invalid notification status response."
                        )
                    notification_status = notification_details.get(
                        "transport_status",
                        notification_status,
                    )
                    notification_error = notification_details.get(
                        "transport_error",
                    )
                except (requests.RequestException, ValueError) as exc:
                    notification_status = "unknown"
                    notification_error = str(exc)

            with self._lock:
                self._status = {
                    "healthy": status,
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
    """Load the models once per Streamlit process."""

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
        "smartvision_webcam_lock_held", False
    ):
        get_webcam_lock().release()

    return True


def current_image_size():
    """Return a valid input size without overriding user settings."""

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


def build_recent_detection_table(recent_detections):
    columns = ["timestamp", "class", "confidence", "track_id"]

    if not isinstance(recent_detections, list):
        return pd.DataFrame(columns=columns)

    rows = [
        {key: item.get(key) for key in columns}
        for item in recent_detections
        if isinstance(item, dict)
    ]

    return pd.DataFrame(rows, columns=columns).tail(20)


def _format_validation_cell(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_format_validation_cell(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if pd.isna(value):
        return "Unavailable"
    return str(value)


def build_validation_acceptance_table(checks):
    columns = ["Check", "Passed", "Actual", "Required"]
    if not isinstance(checks, dict):
        checks = {}

    rows = []
    for name, value in checks.items():
        if not isinstance(value, dict):
            continue
        required = value.get(
            "minimum",
            value.get("maximum", value.get("expected")),
        )
        rows.append({
            "Check": str(name),
            "Passed": value.get("passed") is True,
            "Actual": _format_validation_cell(value.get("actual")),
            "Required": _format_validation_cell(required),
        })

    if not rows:
        return pd.DataFrame({
            "Check": pd.Series(dtype="string"),
            "Passed": pd.Series(dtype="bool"),
            "Actual": pd.Series(dtype="string"),
            "Required": pd.Series(dtype="string"),
        }, columns=columns)

    frame = pd.DataFrame(rows, columns=columns)
    frame["Check"] = frame["Check"].astype("string")
    frame["Passed"] = frame["Passed"].astype(bool)
    frame["Actual"] = frame["Actual"].astype("string")
    frame["Required"] = frame["Required"].astype("string")
    return frame


# ============================================================
# COMMON DASHBOARD HEADER
# ============================================================

def render_status_header(runtime):
    health = get_api_health_monitor().snapshot()
    capture = st.session_state.get("smartvision_capture")
    detector = runtime.get("detector") if runtime else None

    header = st.columns([2, 1, 1, 1, 1, 1])

    header[0].markdown("## 🛡️ SmartVision Pro")

    camera_running = (
        capture is not None and capture.is_running
    )
    camera_open = (
        camera_running and getattr(capture, "opened", False)
    )

    header[1].badge(
        "Camera connected" if camera_open else (
            "Camera recovering" if camera_running else "Camera idle"
        ),
        color="green" if camera_open else (
            "orange" if camera_running else "gray"
        ),
    )

    if detector is None:
        detector_text = "Detector unavailable"
        detector_color = "red"
    elif getattr(detector, "is_custom_model", False):
        if not getattr(detector, "is_validated", False):
            detector_text = "Model validation failed"
            detector_color = "red"
        elif not getattr(detector, "production_alerts_approved", False):
            detector_text = "Model validated · approval missing"
            detector_color = "orange"
        else:
            detector_text = "Model approved"
            detector_color = "green"
    else:
        detector_text = "Generic model"
        detector_color = "orange"

    header[2].badge(detector_text, color=detector_color)

    api_online = health.get("healthy") is True
    header[3].badge(
        "API online" if api_online else (
            "API checking" if health.get("healthy") is None
            else "API offline"
        ),
        color="green" if api_online else (
            "gray" if health.get("healthy") is None else "red"
        ),
    )

    db_status = health.get("database_status", "unknown")
    header[4].badge(
        "Storage online" if db_status == "connected" else (
            "Storage offline" if db_status == "unavailable"
            else "Storage checking"
        ),
        color="green" if db_status == "connected" else (
            "red" if db_status == "unavailable" else "gray"
        ),
    )

    notification_status = health.get(
        "notification_status", "unknown"
    )
    notification_details = health.get("notification_details")
    if (
        isinstance(notification_details, dict)
        and notification_details.get("automatic_alerts_enabled") is True
    ):
        header[5].badge("Automatic alerts enabled", color="green")
    elif notification_status == "ready":
        header[5].badge(
            "ntfy transport ready · auto alerts blocked",
            color="orange",
        )
    else:
        header[5].badge(
            "ntfy transport off" if notification_status == "disabled"
            else "ntfy transport unknown",
            color="gray",
        )


# ============================================================
# LIVE DETECTION PAGE
# ============================================================

@st.fragment(run_every="200ms")
def render_live_inference():
    capture = st.session_state.get("smartvision_capture")
    worker = st.session_state.get("smartvision_worker")
    runtime = init_models_and_helpers()

    if runtime is None:
        st.error("The detector runtime is unavailable.")
        return

    if capture is None or worker is None:
        st.info("Choose a source and press Start.")
        return

    detector = runtime["detector"]

    # Apply the latest settings without restarting the camera.
    worker.update_settings(worker_settings(detector))

    result = worker.snapshot()
    state = result.get("state_data", {})
    health = get_api_health_monitor().snapshot()
    threat_manager = runtime["threat_manager"]
    delivery = threat_manager.get_delivery_health()

    # Model and service status.
    status_cols = st.columns(6)

    status_cols[0].badge(
        "YOLO: validated" if getattr(
            detector, "alerts_enabled", False
        ) else "YOLO: review required",
        color="green" if getattr(
            detector, "alerts_enabled", False
        ) else "orange",
    )

    lstm_ready = runtime["lstm_model"] is not None
    status_cols[1].badge(
        "LSTM: ready" if lstm_ready else "LSTM: unavailable",
        color="green" if lstm_ready else "red",
    )

    pose = worker.pose_estimator
    pose_ready = pose is not None and pose.available

    status_cols[2].badge(
        "MediaPipe: ready" if pose_ready else (
            "MediaPipe: initializing"
            if pose is None and worker.is_running
            else "MediaPipe: unavailable"
        ),
        color="green" if pose_ready else "orange",
    )

    status_cols[3].badge(
        "ByteTrack: running" if worker.is_running
        else "ByteTrack: stopped",
        color="green" if worker.is_running else "gray",
    )

    status_cols[4].badge(
        "API: online" if health["healthy"] is True else (
            "API: offline" if health["healthy"] is False
            else "API: checking"
        ),
        color="green" if health["healthy"] is True else (
            "red" if health["healthy"] is False else "gray"
        ),
    )

    notification_details = health.get("notification_details")
    automatic_alerts_enabled = (
        notification_details.get("automatic_alerts_enabled")
        if isinstance(notification_details, dict)
        else None
    )
    status_cols[5].badge(
        "Automatic alerts: enabled"
        if automatic_alerts_enabled is True
        else "Automatic alerts: blocked"
        if automatic_alerts_enabled is False
        else "Automatic alerts: unknown",
        color="green" if automatic_alerts_enabled is True else "orange",
    )

    if (
        automatic_alerts_enabled is False
        and isinstance(notification_details, dict)
    ):
        st.warning(
            "Automatic notifications are blocked. "
            + str(
                notification_details.get("blocked_reason")
                or "Review the Phone alerts status page."
            )
        )
    elif automatic_alerts_enabled is None and not delivery["enabled"]:
        st.warning(
            "The automatic-alert status endpoint is unavailable and local "
            "notification delivery is disabled. "
            + str(delivery.get("disabled_reason"))
        )

    if not lstm_ready:
        st.info(
            "Action recognition is unavailable until a compatible "
            "LSTM checkpoint is installed."
        )

    if lstm_ready and not pose_ready:
        pose_error = getattr(pose, "diagnostic", None) if pose else None
        st.warning(
            "MediaPipe pose estimation is not ready. "
            + str(pose_error or "Check the pose model configuration.")
        )

    # Display the newest webcam frame, not a delayed queued frame.
    live_frame = None
    live_age_ms = None

    if capture.is_webcam:
        live_result = capture.read(0, count_drops=False)

        if live_result is not None:
            live_frame, sequence, captured_at, _ = live_result
            st.session_state["smartvision_live_sequence"] = sequence
            live_age_ms = max(
                0.0,
                (time.perf_counter() - captured_at) * 1000,
            )

            if live_age_ms > 500:
                live_frame = None
    else:
        live_frame = result.get("last_processed_frame")

    st.subheader("Live camera feed")

    if live_frame is not None:
        st.image(
            live_frame,
            channels="BGR",
            width="stretch",
            alt="Latest live camera frame",
        )
    elif capture.is_webcam:
        st.warning("Waiting for a fresh webcam frame.")
    else:
        st.info("Waiting for the first processed video frame.")

    if capture.error and capture.error != "Video finished.":
        st.error(capture.error)
    elif capture.error == "Video finished.":
        st.info("Video finished.")

    if result.get("error"):
        st.error(f"Inference error: {result['error']}")

    # Current inference results.
    detections = state.get("object_detections", [])
    people = state.get("person_detections", [])
    weapons = [
        item for item in detections
        if item.get("class_id") in (2, 4)
        and item.get("is_project_class", False)
    ]

    st.subheader("Live metrics")
    metrics = st.columns(5)

    metrics[0].metric(
        "Detections",
        len(detections) + len(people),
        border=True,
    )
    metrics[1].metric(
        "People",
        state.get("people_detected", 0),
        border=True,
    )
    metrics[2].metric(
        "Trained weapon detections",
        len(weapons) if getattr(
            detector, "weapon_detection_available", False
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

    if not getattr(detector, "weapon_detection_available", False):
        st.warning(
            "The active model has not been confirmed to support the "
            "project's trained gun/knife classes. Do not rely on it "
            "for real-world security decisions."
        )

    more_metrics = st.columns(4)
    more_metrics[0].metric(
        "Capture FPS",
        f"{capture.capture_fps:.1f}",
        border=True,
    )
    more_metrics[1].metric(
        "Live frame age",
        f"{live_age_ms:.0f} ms"
        if live_age_ms is not None else "Unavailable",
        border=True,
    )
    more_metrics[2].metric(
        "Processing time",
        f"{result.get('processing_ms', 0.0):.0f} ms",
        border=True,
    )
    more_metrics[3].metric(
        "Dropped frames",
        capture.dropped_frames,
        border=True,
    )

    resolution = result.get("capture_resolution")
    inference_resolution = result.get("inference_resolution")

    st.caption(
        f"Capture resolution: {resolution or 'Unavailable'} · "
        f"Inference resolution: "
        f"{inference_resolution or 'Unavailable'} · "
        f"YOLO input setting: {current_image_size()}"
    )

    st.caption(
        "The webcam preview uses the newest captured frame. "
        "Inference may skip frames to prevent a backlog."
    )

    # Stage-level timing.
    timings = result.get("timing_summary_ms", {})

    if timings:
        st.subheader("Inference performance breakdown")

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

        visible_stages = [
            stage for stage in stages if stage in timings
        ]

        for start in range(0, len(visible_stages), 4):
            stage_cols = st.columns(
                min(4, len(visible_stages) - start)
            )

            for column, stage in zip(
                stage_cols,
                visible_stages[start:start + 4],
            ):
                summary = timings[stage]
                column.metric(
                    stage.replace("_", " ").title(),
                    f"{summary.get('mean', 0.0):.1f} ms",
                    delta=(
                        f"p95 {summary.get('p95', 0.0):.1f} ms"
                    ),
                )

    # Recent detections.
    st.subheader("Recent detections")
    recent = build_recent_detection_table(
        result.get("recent_detections", [])
    )

    if recent.empty:
        st.caption("No detections reported yet.")
    else:
        st.dataframe(
            recent,
            hide_index=True,
            width="stretch",
        )

    # LSTM action results.
    st.subheader("Action-recognition results")
    actions = state.get("action_detections", [])

    if actions:
        for action in actions:
            st.write(
                f"{action.get('class', 'Unknown')} · "
                f"{action.get('confidence', 0.0):.1%} · "
                f"Track ID {action.get('track_id', 'Unknown')}"
            )
    elif not lstm_ready:
        st.warning("Action recognition is unavailable.")
    else:
        st.caption("No non-normal action reported for the latest sequence.")

    # Session incident history.
    with st.expander("Incident history and delivery status"):
        incidents = threat_manager.get_incident_history(limit=10)

        if not incidents:
            st.caption("No incidents have been raised in this session.")

        for incident in incidents:
            st.write(
                f"{incident.get('created_at')} · "
                f"{incident.get('event_type')} · "
                f"{incident.get('label')} · "
                f"{incident.get('confidence', 0.0):.1%} · "
                f"{incident.get('status')}"
            )

            if incident.get("snapshot"):
                st.image(
                    incident["snapshot"],
                    caption="Incident snapshot",
                    width=320,
                )

            if incident.get("error"):
                st.caption(incident["error"])

        st.caption(
            f"Alerts waiting to send: {delivery.get('queued', 0)}"
        )


@st.fragment(run_every="1s")
def render_inference_snapshot():
    worker = st.session_state.get("smartvision_worker")

    if worker is None:
        return

    result = worker.snapshot()
    frame = result.get("last_processed_frame")

    st.subheader("Latest annotated inference")

    if frame is None:
        st.caption("No completed inference frame is available yet.")
        return

    last_processed_at = result.get("last_processed_at")
    if last_processed_at:
        age_ms = max(
            0.0,
            (time.perf_counter() - last_processed_at) * 1000,
        )
        st.caption(
            f"Last completed inference: {age_ms:.0f} ms ago."
        )

    st.image(
        frame,
        channels="BGR",
        width="stretch",
        alt="Completed inference frame with detection annotations",
    )


@st.fragment(run_every="1s")
def render_video_progress():
    capture = st.session_state.get("smartvision_capture")
    worker = st.session_state.get("smartvision_worker")
    if capture is None or worker is None:
        return
    frame_count = getattr(capture, "frame_count", 0)
    if frame_count > 0:
        processed_frames = worker.snapshot().get("processed_frames", 0)
        progress = min(1.0, processed_frames / frame_count)
        st.progress(
            progress,
            text=(
                f"Processed {min(processed_frames, frame_count):,} "
                f"of {frame_count:,} video frames"
            ),
        )
    else:
        st.caption(
            "Video frame count is unavailable; processing continues "
            "through the bounded inference worker."
        )


# ============================================================
# LIVE DETECTION CONTROLS
# ============================================================

def run_streamlit_inference_page(video_analysis=False):
    if video_analysis:
        st.title("Video analysis")
        st.caption(
            "Analyze an uploaded or configured video using the same "
            "inference worker as live detection."
        )
        source_key = "smartvision_video_source_choice"
        default_source = "Upload Video"
    else:
        st.title("Live detection")
        st.caption(
            "YOLO object detection · ByteTrack tracking · "
            "LSTM action recognition · MediaPipe pose estimation"
        )
        source_key = "smartvision_live_source_choice"
        default_source = "Live Webcam"

    runtime = init_models_and_helpers()

    if runtime is None or runtime.get("detector") is None:
        st.error("YOLO model could not be loaded. Check the model path.")
        return

    detector = runtime["detector"]

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

    class_ids = available_class_ids(detector)
    class_labels = [
        label for class_id, label in (
            (5, "Person"),
            (2, "Firearm"),
            (4, "Knife/blade"),
        )
        if class_id in class_ids
    ]

    st.sidebar.caption(
        "Available safety classes: "
        + (", ".join(class_labels) if class_labels else "None")
    )

    if st.sidebar.button("Reload model and settings"):
        if not stop_active_source():
            st.error(
                "The worker did not stop cleanly. "
                "Stop the active source before reloading."
            )
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
    uploaded_temp_path = None

    if input_option == "Live Webcam":
        source_path = int(
            st.session_state.get(
                "smartvision_camera_index",
                WEBCAM_ID,
            )
        )
        st.info("Live webcam selected. Press Start to begin.")

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
                    "Choose a video",
                    files,
                )
                source_path = os.path.join(
                    INPUT_VIDEO_DIR,
                    selected,
                )
                st.info(f"Selected video: {selected}")
            else:
                st.warning(
                    f"No supported videos found in {INPUT_VIDEO_DIR}"
                )
        else:
            st.error(f"Input folder not found: {INPUT_VIDEO_DIR}")

    elif input_option == "Upload Video":
        uploaded = st.sidebar.file_uploader(
            "Choose a video",
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
                    uploaded_temp_path = old_path
                else:
                    if old_path and os.path.isfile(old_path):
                        os.remove(old_path)

                    suffix = os.path.splitext(uploaded.name)[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_file:
                        temp_file.write(uploaded.getbuffer())
                        uploaded_temp_path = temp_file.name

                    st.session_state[
                        "smartvision_uploaded_temp_path"
                    ] = uploaded_temp_path
                    st.session_state[
                        "smartvision_uploaded_signature"
                    ] = signature

                source_path = uploaded_temp_path
                st.success("Video is ready.")

            except OSError as exc:
                st.error(f"Could not save the uploaded video: {exc}")

    start_col, stop_col = st.sidebar.columns(2)

    if start_col.button(
        "Start",
        type="primary",
        key="start_smartvision",
    ):
        if source_path is None:
            st.error("Select a valid video source first.")

        else:
            old_capture = st.session_state.get("smartvision_capture")
            old_worker = st.session_state.get("smartvision_worker")

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
                st.error(
                    "The previous worker is still stopping. "
                    "Wait and press Stop again."
                )

            else:
                lock = get_webcam_lock()
                lock_acquired = False

                if isinstance(source_path, int):
                    lock_acquired = lock.acquire(blocking=False)

                    if not lock_acquired:
                        st.error(
                            "The webcam is already in use by another "
                            "SmartVision session."
                        )

                if (
                    not isinstance(source_path, int)
                    or lock_acquired
                ):
                    if lock_acquired:
                        st.session_state[
                            "smartvision_webcam_lock_held"
                        ] = True

                    capture = LatestFrameCapture(source_path)

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
                                f"Could not start inference worker: {exc}"
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
                            capture.error or "Could not open the source."
                        )
                        capture.stop()

                        if lock_acquired:
                            st.session_state.pop(
                                "smartvision_webcam_lock_held",
                                None,
                            )
                            lock.release()

    if stop_col.button("Stop", key="stop_smartvision"):
        if stop_active_source():
            uploaded_path = st.session_state.pop(
                "smartvision_uploaded_temp_path",
                None,
            )

            if uploaded_path and os.path.isfile(uploaded_path):
                try:
                    os.remove(uploaded_path)
                except OSError as exc:
                    st.warning(
                        f"Temporary file cleanup failed: {exc}"
                    )

            st.session_state.pop(
                "smartvision_uploaded_signature",
                None,
            )
            st.success("Inference stopped.")
        else:
            st.error(
                "The source has not stopped completely. "
                "Wait briefly and press Stop again."
            )

    capture = st.session_state.get("smartvision_capture")

    if capture is not None:
        if video_analysis and not capture.is_webcam:
            render_video_progress()
        render_live_inference()
        render_inference_snapshot()
    else:
        st.info("No active source. Select a source and press Start.")


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
            return [], f"Incident API returned HTTP {response.status_code}."

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
            raise ValueError("Summary response is not an object.")

        if (
            type(summary.get("total")) is not int
            or type(summary.get("today")) is not int
            or not isinstance(summary.get("by_class"), dict)
        ):
            raise ValueError("Summary response has a malformed structure.")

        return summary, None

    except (requests.RequestException, ValueError, TypeError) as exc:
        return None, f"Could not load incident summary: {exc}"


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
            pd.to_datetime(str(created_at), errors="coerce", utc=True)
            if created_at is not None
            else pd.NaT
        )

        row["threat_level"] = str(
            row.get("threat_level") or "Unknown"
        )
        row["label"] = str(row.get("label") or "Unknown")
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
                    "detail", "Notification test failed."
                )
            except (ValueError, AttributeError):
                detail = "Notification test failed."

            return False, str(detail)

        result = response.json()

        if (
            not isinstance(result, dict)
            or result.get("status") != "sent"
        ):
            return False, "Backend did not confirm notification delivery."

        return True, result.get(
            "message", "Test notification accepted."
        )

    except (requests.RequestException, ValueError, TypeError) as exc:
        return False, f"Notification test failed: {exc}"


def read_json_report(path):
    try:
        with Path(path).open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except FileNotFoundError:
        return None, f"Report is unavailable: {Path(path)}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Could not read report: {exc}"
    if not isinstance(report, dict):
        return None, f"Report has an invalid structure: {Path(path)}"
    return report, None


def latest_dataset_audit():
    if not DATASET_AUDIT_ROOT.is_dir():
        return None
    reports = sorted(
        DATASET_AUDIT_ROOT.rglob("dataset_audit.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def render_model_performance():
    st.title("Model performance")
    report, error = read_json_report(VALIDATION_REPORT_PATH)
    if error:
        st.info(
            "No completed validation report is available yet. Run the "
            "separate detector evaluator; the dashboard does not infer "
            "validation results from checkpoint presence."
        )
        st.caption(error)
        return
    if report is None:
        st.error("The validation report could not be loaded.")
        return

    summary = report.get("validation_summary", {})
    required_metrics = ("precision", "recall", "map50", "map50_95")
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
    representative_record = report.get("representative_tests")
    passed = (
        report.get("validation_completed") is True
        and report.get("validation_passed") is True
        and report.get("evaluation_split") == "val"
        and metrics_present
        and checks_passed
        and isinstance(representative_record, dict)
        and representative_record.get("completed") is True
    )
    if passed:
        st.success("The recorded validation passed its acceptance checks.")
    else:
        st.error(
            "Validation did not pass. Production approval and automatic "
            "alerts must remain disabled."
        )

    st.caption(
        f"Checkpoint: {report.get('checkpoint_path', 'Unavailable')} · "
        f"SHA-256: {report.get('checkpoint_sha256', 'Unavailable')}"
    )
    st.caption(
        f"Completed: {report.get('validation_completed_at', 'Unavailable')} · "
        f"Split: {report.get('evaluation_split', 'Unavailable')}"
    )
    metric_columns = st.columns(4)
    for column, name, label in zip(
        metric_columns,
        ("precision", "recall", "map50", "map50_95"),
        ("Precision", "Recall", "mAP50", "mAP50-95"),
    ):
        value = summary.get(name)
        column.metric(
            label,
            f"{value:.3f}" if isinstance(value, (int, float)) else "Unavailable",
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
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    checks = acceptance_checks
    if isinstance(checks, dict) and checks:
        st.subheader("Acceptance checks")
        st.dataframe(
            build_validation_acceptance_table(checks),
            hide_index=True,
            width="stretch",
        )

    representative = report.get("representative_tests", {})
    if isinstance(representative, dict):
        st.subheader("Representative labeled-image tests")
        st.write(
            "Completed:",
            representative.get("completed") is True,
            "· Classes tested:",
            ", ".join(representative.get("tested_classes", []))
            or "Unavailable",
        )
        st.write(
            "Missed weapon detections:",
            representative.get("missed_weapon_detections", "Unavailable"),
            "· Weapon false positives:",
            representative.get("weapon_false_positives", "Unavailable"),
        )

    confusion_path = report.get("confusion_matrix_image")
    if confusion_path and Path(confusion_path).is_file():
        st.image(
            str(confusion_path),
            caption="Confusion matrix from this validation run",
            width="stretch",
        )
    else:
        st.caption("Confusion-matrix visualization is unavailable.")
    for name, check in (checks or {}).items():
        if isinstance(check, dict) and check.get("passed") is not True:
            st.warning(f"Failed check: {name}")


def render_dataset_training():
    st.title("Dataset and training")
    audit_path = latest_dataset_audit()
    if audit_path is None:
        st.info(
            "No dataset audit report is available. Run the read-only audit "
            "before starting a training experiment."
        )
    else:
        audit, error = read_json_report(audit_path)
        if error:
            st.warning(error)
        elif audit is not None:
            if audit.get("audit_passed") is True:
                st.success("Latest recorded dataset audit passed.")
            else:
                st.error("Latest recorded dataset audit did not pass.")
            st.caption(f"Report: {audit_path}")
            st.caption(
                f"Dataset YAML: {audit.get('dataset_yaml', 'Unavailable')} · "
                f"Completed: {audit.get('audit_completed_at', 'Unavailable')}"
            )
            split_summary = []
            for split_name, values in audit.get("splits", {}).items():
                class_rows = values.get("classes", {})
                valid_annotations = values.get(
                    "valid_annotation_rows",
                    sum(
                        counts.get("instances", 0)
                        for counts in class_rows.values()
                        if isinstance(counts, dict)
                    ),
                )
                valid_labels = values.get(
                    "valid_label_files",
                    values.get("label_files_found", 0)
                    - len(values.get("invalid_annotation_files", [])),
                )
                split_summary.append({
                    "Split": split_name,
                    "Images": values.get("image_count", "Unavailable"),
                    "Readable": values.get("readable_images", "Unavailable"),
                    "Labels": values.get("label_files_found", "Unavailable"),
                    "Valid labels": valid_labels,
                    "Valid annotations": valid_annotations,
                    "Invalid rows": values.get(
                        "invalid_annotation_rows",
                        "Unavailable",
                    ),
                })
            if split_summary:
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
            duplicates = audit.get("duplicates", {})
            if isinstance(duplicates, dict):
                st.write(
                    "Exact cross-split image leakage groups:",
                    len(duplicates.get("exact_cross_split_leakage", [])),
                )
                st.write(
                    "Same-name cross-split review candidates:",
                    len(duplicates.get("same_basename_cross_split_suspicions", [])),
                )
            warnings = audit.get("warnings", [])
            for warning in warnings:
                st.warning(warning)
            visualizations = audit.get("visualizations", [])
            if visualizations:
                st.subheader("Parsed annotation samples")
                for item in visualizations[:6]:
                    image_path = item.get("visualization")
                    if image_path and Path(image_path).is_file():
                        st.image(
                            str(image_path),
                            caption=f"{item.get('split')} · {item.get('class')}",
                            width=260,
                        )

    st.subheader("Training experiments")
    runs_root = Path(ROOT_DIR) / "runs" / "detect"
    run_directories = sorted(
        (
            path for path in runs_root.glob("smartvision_weapon_*")
            if path.is_dir()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if runs_root.is_dir() else []
    if not run_directories:
        st.info("No SmartVision training experiments are recorded.")
    else:
        run_rows = []
        for run_directory in run_directories:
            completion_path = (
                run_directory / "smartvision_training_completion.json"
            )
            run_rows.append({
                "Run": run_directory.name,
                "Completed": completion_path.is_file(),
                "Best checkpoint": (
                    str(run_directory / "weights" / "best.pt")
                    if (run_directory / "weights" / "best.pt").is_file()
                    else "Unavailable"
                ),
            })
        st.dataframe(pd.DataFrame(run_rows), hide_index=True, width="stretch")

    st.code(
        r".\.venv\Scripts\python.exe scripts\train_weapon_improved.py "
        r"--data The_final-4\data.yaml",
        language="powershell",
    )
    st.caption(
        "Training is a separate, resource-intensive operation. This page "
        "does not start or resume a training job."
    )


def render_analytics():
    st.title("Analytics")
    st.caption(
        "Charts below use actual stored incident records returned by the "
        "backend; the incident API query is capped at 200 records."
    )
    summary, summary_error = fetch_incident_summary()
    incidents, incident_error = fetch_incidents(limit=200)
    if summary_error:
        st.warning(summary_error)
    if incident_error:
        st.error(incident_error)
        return
    if summary is not None:
        columns = st.columns(3)
        columns[0].metric("All-time incidents", summary["total"])
        columns[1].metric("Incidents today", summary["today"])
        columns[2].metric("Recorded classes", len(summary["by_class"]))
    frame, invalid = normalize_incidents(incidents)
    if invalid:
        st.warning(f"Ignored {invalid} invalid API records.")
    if frame.empty:
        st.info("No stored incident data is available for analytics.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Incidents by class")
        st.bar_chart(frame["label"].value_counts(), horizontal=True)
    with right:
        st.subheader("Notification delivery outcomes")
        st.bar_chart(
            frame["notification_status"].value_counts(),
            horizontal=True,
        )

    timed = frame.dropna(subset=["created_at"])
    if not timed.empty:
        st.subheader("Incident activity over time")
        st.line_chart(
            timed.groupby(pd.Grouper(key="created_at", freq="h")).size()
        )


def render_phone_alerts():
    st.title("Phone alerts")
    health = get_api_health_monitor().snapshot()
    details = health.get("notification_details")
    if not isinstance(details, dict):
        st.warning(
            "Notification status is unavailable until the FastAPI status "
            "endpoint responds."
        )
        if health.get("notification_error"):
            st.caption(health["notification_error"])
        return

    values = (
        ("Transport", details.get("transport_status", "unknown")),
        (
            "Notification configuration",
            "Enabled" if details.get("notification_configuration_enabled")
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
            "Production alert approval",
            "Granted" if details.get("production_alerts_approved") else "Missing",
        ),
        (
            "Automatic alerts",
            "Enabled" if details.get("automatic_alerts_enabled") else "Blocked",
        ),
    )
    st.dataframe(
        pd.DataFrame([{"Gate": label, "Status": value} for label, value in values]),
        hide_index=True,
        width="stretch",
    )
    blocked_reason = details.get("blocked_reason")
    if blocked_reason:
        st.error(f"Automatic alerts are blocked: {blocked_reason}")

    failures = details.get("validation_failures", [])
    if failures:
        st.subheader("Validation blockers")
        for failure in failures:
            st.warning(str(failure))

    if details.get("transport_status") == "ready":
        st.subheader("Manual transport test")
        st.caption(
            "This sends a test request only. Provider acceptance does not "
            "confirm delivery to the phone."
        )
        if st.button("Send test request to ntfy", type="primary"):
            sent, message = send_test_notification()
            if sent:
                st.success(
                    "ntfy accepted the test request. Phone receipt is not "
                    f"confirmed. {message}"
                )
            else:
                st.error(message)

    st.subheader("Recent incident delivery attempts")
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
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No incident delivery attempts are recorded.")

    st.info(
        "Operator approval is a separate, explicit step. Review the matching "
        "validation report, configure the authorized Windows account, and "
        "run scripts\\approve_detector_alerts.py only after all checks pass."
    )


# ============================================================
# OVERVIEW PAGE
# ============================================================

@st.fragment(run_every="10s")
def render_overview():
    st.title("Overview")
    st.caption("SmartVision Pro system summary")

    summary, summary_error = fetch_incident_summary()
    capture = st.session_state.get("smartvision_capture")
    worker = st.session_state.get("smartvision_worker")

    result = worker.snapshot() if worker is not None else {}
    state = result.get("state_data", {})

    weapon_items = state.get("weapon_detections", [])

    cols = st.columns(7)

    cols[0].metric(
        "Active cameras",
        int(
            capture is not None
            and capture.is_webcam
            and capture.is_running
        ),
        border=True,
    )
    cols[1].metric(
        "People detected",
        state.get("people_detected", 0) if worker is not None else "Unavailable",
        border=True,
    )
    cols[2].metric(
        "Weapon detections",
        len(weapon_items) if worker is not None else "Unavailable",
        border=True,
    )
    cols[3].metric(
        "Total incidents",
        summary["total"] if summary else "Unavailable",
        border=True,
    )
    cols[4].metric(
        "Today's incidents",
        summary["today"] if summary else "Unavailable",
        border=True,
    )
    cols[5].metric(
        "Inference FPS",
        f"{result['inference_fps']:.1f}" if worker is not None else "Unavailable",
        border=True,
    )
    yolo_timing = result.get("timing_summary_ms", {}).get("yolo", {})
    yolo_latency = yolo_timing.get("mean") if isinstance(yolo_timing, dict) else None
    cols[6].metric(
        "YOLO latency",
        f"{yolo_latency:.1f} ms"
        if isinstance(yolo_latency, (int, float))
        else "Unavailable",
        border=True,
    )

    if summary_error:
        st.warning(summary_error)

    notification_details = get_api_health_monitor().snapshot().get(
        "notification_details"
    )
    if isinstance(notification_details, dict) and notification_details.get(
        "blocked_reason"
    ):
        st.warning(
            "Automatic notifications are blocked: "
            + str(notification_details["blocked_reason"])
        )

    alerts, error = fetch_incidents(limit=200)

    if error:
        st.warning(error)
        return

    valid = [row for row in alerts if isinstance(row, dict)]

    if not valid:
        st.info("No stored incidents are available yet.")
    else:
        frame = pd.DataFrame(valid)
        left, right = st.columns(2)

        with left:
            st.subheader("Incidents by severity")
            if "threat_level" in frame:
                st.bar_chart(
                    frame["threat_level"].fillna("Unknown").value_counts(),
                    horizontal=True,
                )

        with right:
            st.subheader("Incidents by class")
            if "label" in frame:
                st.bar_chart(
                    frame["label"].fillna("Unknown").value_counts(),
                    horizontal=True,
                )

        if "created_at" in frame:
            frame["created_at"] = pd.to_datetime(
                frame["created_at"],
                errors="coerce",
                utc=True,
            )
            timed = frame.dropna(subset=["created_at"])

            if not timed.empty:
                st.subheader("Incident activity over time")
                hourly = timed.groupby(
                    pd.Grouper(key="created_at", freq="h")
                ).size()
                st.line_chart(hourly)


# ============================================================
# ALERT HISTORY PAGE
# ============================================================

@st.fragment(run_every="30s")
def render_alert_history():
    st.title("Alert history")
    st.caption("Stored incidents returned by the FastAPI backend")

    summary, summary_error = fetch_incident_summary()

    if summary_error:
        st.warning(summary_error)

    if summary:
        cols = st.columns(3)
        cols[0].metric("Total incidents", summary["total"])
        cols[1].metric("Today's incidents", summary["today"])
        cols[2].metric(
            "Classes recorded",
            len(summary.get("by_class", {})),
        )

        if summary.get("by_class"):
            st.subheader("Incidents by class")
            st.dataframe(
                pd.DataFrame([
                    {"Class": name, "Count": count}
                    for name, count in summary["by_class"].items()
                ]),
                hide_index=True,
                width="stretch",
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
        st.warning(f"Ignored {invalid_count} malformed records.")

    if df.empty:
        st.info("No valid incident records were returned.")
        return

    search = st.text_input(
        "Search",
        placeholder="Class, camera, or severity",
    )

    severity_options = sorted(
        df["threat_level"].dropna().unique().tolist()
    )
    class_options = sorted(
        df["label"].dropna().unique().tolist()
    )

    selected_severity = st.multiselect(
        "Severity",
        severity_options,
        default=severity_options,
    )
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
        "Export incidents as CSV",
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
                        else os.path.join(ROOT_DIR, image_path)
                    )

                    try:
                        safe_path = (
                            os.path.commonpath([
                                os.path.abspath(ROOT_DIR),
                                os.path.abspath(candidate),
                            ]) == os.path.abspath(ROOT_DIR)
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
                st.write("Incident ID:", incident.get("id", "Unknown"))
                st.write("Camera:", incident.get("camera_id", "Unknown"))
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
# CAMERA MANAGEMENT PAGE
# ============================================================

def render_camera_management():
    st.title("Camera management")

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

    capture = st.session_state.get("smartvision_capture")

    with st.container(border=True):
        st.subheader("Camera status")

        if capture is not None and capture.is_running:
            st.success("Camera capture is running.")
            st.write(
                "Capture FPS:",
                f"{capture.capture_fps:.1f}",
            )
            st.write(
                "Capture resolution:",
                getattr(capture, "resolution", "See live metrics"),
            )
        else:
            st.info("No camera is currently streaming.")

    st.caption(
        "Select Live Webcam under Live detection and press Start "
        "to connect to the configured camera."
    )


# ============================================================
# DETECTION SETTINGS PAGE
# ============================================================

def render_detection_settings(runtime):
    st.title("Detection settings")
    st.caption("Adjust inference speed and confidence thresholds.")

    detector = runtime["detector"]
    class_ids = available_class_ids(detector)

    widget_key = "smartvision_yolo_input_size"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = current_image_size()
    elif st.session_state[widget_key] not in IMAGE_SIZE_OPTIONS:
        st.session_state[widget_key] = 320

    st.select_slider(
        "YOLO input size",
        options=IMAGE_SIZE_OPTIONS,
        key=widget_key,
        help=(
            "320 is the recommended starting point for CPU webcam "
            "inference. Larger sizes may improve small-object detail "
            "but increase processing time."
        ),
    )

    st.session_state["smartvision_image_size"] = int(
        st.session_state[widget_key]
    )

    st.caption(
        f"Selected inference size: "
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
        item for item in options if item[0] in class_ids
    ]

    enabled = set(
        st.session_state.get(
            "smartvision_enabled_classes",
            class_ids,
        )
    )

    for class_id, label in options:
        value = st.checkbox(
            f"Enable {label} detection",
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

    st.session_state["smartvision_confidence_threshold"] = st.slider(
        "Global minimum confidence",
        min_value=0.05,
        max_value=0.99,
        value=min(0.99, max(0.05, global_threshold)),
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
            thresholds.get(class_id, CONFIDENCE_THRESHOLD)
        )

        thresholds[class_id] = st.slider(
            f"{label} confidence",
            min_value=0.05,
            max_value=0.99,
            value=min(0.99, max(0.05, threshold)),
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
        step=1,
    )

    st.info(
        "Changing the input size affects inference speed and may affect "
        "small-object detection. Validate accuracy at the selected size."
    )


# ============================================================
# MODEL AND SYSTEM HEALTH PAGE
# ============================================================

def render_system_health(runtime):
    st.title("Model and system health")

    st.caption(f"Python {sys.version.split()[0]} · PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        st.info(
            f"CUDA is available with {torch.cuda.device_count()} device(s)."
        )
    else:
        st.info("CUDA is unavailable; PyTorch inference uses CPU.")

    if runtime is None or runtime.get("detector") is None:
        st.error("Detector is unavailable.")
        return

    detector = runtime["detector"]
    health = get_api_health_monitor().snapshot()

    columns = st.columns(3)
    columns[0].metric("Device", str(runtime["device"]))
    columns[1].metric(
        "YOLO input size",
        current_image_size(),
    )
    columns[2].metric(
        "API response time",
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
            "A generic model is active. Do not treat its results as "
            "validated weapon detection."
        )

    st.write(
        "Production alerts enabled:",
        bool(getattr(detector, "alerts_enabled", False)),
    )
    st.write(
        "Weapon classes available:",
        bool(
            getattr(detector, "weapon_detection_available", False)
        ),
    )

    names = detector.class_names or {}
    class_names = (
        list(names.values()) if isinstance(names, dict)
        else list(names)
    )
    st.write("Model classes:", ", ".join(map(str, class_names)))

    st.subheader("Action recognition and pose")

    st.write(
        "LSTM:",
        "Loaded" if runtime["lstm_model"] is not None
        else "Unavailable",
    )

    worker = st.session_state.get("smartvision_worker")
    pose = worker.pose_estimator if worker is not None else None

    if runtime["lstm_model"] is None:
        st.info(
            "Install a compatible LSTM checkpoint to enable "
            "action recognition."
        )
    elif pose is not None and pose.available:
        st.success("MediaPipe pose estimation is available.")
    else:
        st.warning(
            getattr(
                pose,
                "diagnostic",
                "MediaPipe has not been initialized by a running worker.",
            )
        )

    st.write(
        "ByteTrack:",
        "Running" if worker is not None and worker.is_running
        else "Not running",
    )

    st.subheader("FastAPI and storage")

    st.write(
        "Backend:",
        "Online" if health["healthy"] is True else "Unavailable",
    )
    st.write(
        "Database:",
        health.get("database_status", "unknown"),
    )
    st.write(
        "Phone notifications:",
        health.get("notification_status", "unknown"),
    )

    notification_details = health.get("notification_details")
    if isinstance(notification_details, dict):
        st.subheader("Production notification gates")
        st.write(
            "Notification configuration enabled:",
            bool(notification_details.get("notification_configuration_enabled")),
        )
        st.write(
            "Backend custom detector loaded:",
            bool(notification_details.get("custom_model_loaded")),
        )
        st.write(
            "Detector validation passed:",
            bool(notification_details.get("model_validation_passed")),
        )
        st.write(
            "Production alert approval granted:",
            bool(notification_details.get("production_alerts_approved")),
        )
        st.write(
            "Automatic alerts enabled:",
            bool(notification_details.get("automatic_alerts_enabled")),
        )
        if notification_details.get("blocked_reason"):
            st.warning(notification_details["blocked_reason"])

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

    st.write("Pending alert queue:", delivery.get("queued", 0))

    if not delivery.get("enabled"):
        st.warning(
            "Alert delivery is disabled: "
            + str(delivery.get("disabled_reason"))
        )


# ============================================================
# MAIN NAVIGATION
# ============================================================

def main():
    runtime = init_models_and_helpers()
    render_status_header(runtime)

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Live detection",
            "Video analysis",
            "Incident history",
            "Analytics",
            "Model performance",
            "Dataset and training",
            "Phone alerts",
            "System health",
            "Settings",
        ],
        key="smartvision_navigation",
    )

    if page == "Overview":
        render_overview()

    elif page == "Live detection":
        run_streamlit_inference_page()

    elif page == "Video analysis":
        run_streamlit_inference_page(video_analysis=True)

    elif page == "Incident history":
        render_alert_history()

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
            st.error("The detector could not be loaded.")
        else:
            render_detection_settings(runtime)


if __name__ == "__main__":
    main()
