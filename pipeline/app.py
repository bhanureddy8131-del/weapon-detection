
import os
import sys
import time
import tempfile
import threading
from datetime import date, datetime, time as datetime_time

import pandas as pd
import requests
import streamlit as st

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
TEST_NOTIFICATION_URL = f"{API_BASE_URL}/notifications/test"

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

            with self._lock:
                self._status = {
                    "healthy": status,
                    "database_status": database_status,
                    "database_error": database_error,
                    "notification_status": notification_status,
                    "notification_error": notification_error,
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
    elif getattr(detector, "alerts_enabled", False):
        detector_text = "Threat model approved"
        detector_color = "green"
    elif getattr(detector, "is_custom_model", False):
        detector_text = "Custom model unvalidated"
        detector_color = "orange"
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
    header[5].badge(
        "Phone alerts ready" if notification_status == "ready"
        else "Phone alerts off" if notification_status == "disabled"
        else "Phone alerts unknown",
        color="green" if notification_status == "ready"
        else "gray",
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

    status_cols[5].badge(
        "Alerts: enabled" if delivery["enabled"]
        else "Alerts: disabled",
        color="green" if delivery["enabled"] else "orange",
    )

    if not delivery["enabled"]:
        st.warning(
            "Threat notifications are disabled. "
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


# ============================================================
# LIVE DETECTION CONTROLS
# ============================================================

def run_streamlit_inference_page():
    st.title("Live detection")
    st.caption(
        "YOLO object detection · ByteTrack tracking · "
        "LSTM action recognition · MediaPipe pose estimation"
    )

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

    input_option = st.sidebar.selectbox(
        "Input source",
        [
            "Live Webcam",
            "Select from Input Folder",
            "Upload Video",
        ],
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
            raise ValueError("Summary response has an invalid structure.")

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

        row["created_at"] = pd.to_datetime(
            row.get("created_at"),
            errors="coerce",
            utc=True,
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

    cols = st.columns(6)

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
        state.get("people_detected", 0),
        border=True,
    )
    cols[2].metric(
        "Weapon detections",
        len(weapon_items),
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
        f"{result.get('inference_fps', 0.0):.1f}",
        border=True,
    )

    if summary_error:
        st.warning(summary_error)

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

    # Use a persistent state key so Streamlit preserves the selection.
    if "smartvision_yolo_input_size" not in st.session_state:
        initial_size = current_image_size()
        st.session_state["smartvision_yolo_input_size"] = initial_size

    current_size = st.session_state["smartvision_yolo_input_size"]

    if current_size not in IMAGE_SIZE_OPTIONS:
        current_size = 320
        st.session_state["smartvision_yolo_input_size"] = 320

    st.select_slider(
        "YOLO input size",
        options=IMAGE_SIZE_OPTIONS,
        value=current_size,
        key="smartvision_yolo_input_size",
        help=(
            "320 is the recommended starting point for CPU webcam "
            "inference. Larger sizes may improve small-object detail "
            "but increase processing time."
        ),
    )

    st.session_state["smartvision_image_size"] = int(
        st.session_state["smartvision_yolo_input_size"]
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
            "Alert history",
            "Camera management",
            "Detection settings",
            "Model and system health",
        ],
    )

    if page == "Overview":
        render_overview()

    elif page == "Live detection":
        run_streamlit_inference_page()

    elif page == "Alert history":
        render_alert_history()

    elif page == "Camera management":
        render_camera_management()

    elif page == "Detection settings":
        if runtime is None:
            st.error("The detector could not be loaded.")
        else:
            render_detection_settings(runtime)

    elif page == "Model and system health":
        render_system_health(runtime)


if __name__ == "__main__":
    main()
