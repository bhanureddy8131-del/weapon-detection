
import os
import sys
import time
import tempfile
import threading
from datetime import date, datetime, time as datetime_time

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------
# PROJECT PATHS AND IMPORTS
# --------------------------------------------------

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

try:
    from pipeline.run_inference import (
        get_canonical_class_id,
        load_models,
    )
except ImportError:
    from run_inference import get_canonical_class_id, load_models

from pipeline.inference_worker import InferenceWorker
from pipeline.latest_frame_capture import LatestFrameCapture


# --------------------------------------------------
# STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartVision Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = f"{API_BASE_URL}/incidents"


class APIHealthMonitor:
    """Probe the backend outside Streamlit's UI and inference threads."""

    def __init__(self, url, interval_seconds=5.0):
        self.url = url
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._status = {
            "healthy": None,
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
            started_at = time.perf_counter()
            try:
                response = requests.get(
                    self.url,
                    params={"limit": 1},
                    timeout=(0.5, 1.0),
                )
                response.raise_for_status()
                healthy = True
                error = None
            except requests.RequestException as exc:
                healthy = False
                error = str(exc)

            with self._lock:
                self._status = {
                    "healthy": healthy,
                    "latency_ms": (time.perf_counter() - started_at) * 1000,
                    "error": error,
                    "checked_at": datetime.now(),
                }
            self._stop_event.wait(self.interval_seconds)

    def snapshot(self):
        with self._lock:
            return dict(self._status)


@st.cache_resource
def get_api_health_monitor():
    return APIHealthMonitor(API_URL)


@st.cache_resource
def get_webcam_lock():
    """Allow only one Streamlit session per process to own the webcam."""
    return threading.Lock()


def stop_active_source():
    worker = st.session_state.get("smartvision_worker")
    capture = st.session_state.get("smartvision_capture")

    worker_stopped = worker is None or worker.stop()
    capture_stopped = capture is None or capture.stop()
    if not worker_stopped or not capture_stopped:
        return False

    st.session_state.pop("smartvision_worker", None)
    st.session_state.pop("smartvision_capture", None)
    st.session_state.pop("smartvision_live_sequence", None)
    st.session_state.pop("smartvision_last_live_frame", None)
    st.session_state.pop("smartvision_last_live_at", None)

    if st.session_state.pop("smartvision_webcam_lock_held", False):
        get_webcam_lock().release()
    return True


# --------------------------------------------------
# LOAD AND CACHE MODELS
# --------------------------------------------------

@st.cache_resource
def init_models_and_helpers():
    """Load detection models and reusable inference helpers once."""

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


def _available_class_ids(detector):
    names = detector.class_names or {}
    class_ids = names.keys() if isinstance(names, dict) else range(len(names))
    return {
        get_canonical_class_id(detector, int(raw_class_id))
        for raw_class_id in class_ids
    }


@st.fragment(run_every="100ms")
def render_live_inference():
    """Render fresh capture frames while a separate worker performs inference."""
    capture = st.session_state.get("smartvision_capture")
    worker = st.session_state.get("smartvision_worker")
    runtime = init_models_and_helpers()

    if runtime is None:
        st.error("The detector runtime is unavailable.")
        return
    if capture is None or worker is None:
        st.info("Select a source and start inference.")
        return

    worker.update_settings({
        "confidence_threshold": st.session_state.get(
            "smartvision_confidence_threshold",
            CONFIDENCE_THRESHOLD,
        ),
        "image_size": st.session_state.get(
            "smartvision_image_size",
            YOLO_IMAGE_SIZE,
        ),
        "class_thresholds": st.session_state.get(
            "smartvision_class_thresholds",
            CLASS_CONFIDENCE_THRESHOLDS,
        ),
        "enabled_classes": st.session_state.get(
            "smartvision_enabled_classes",
            _available_class_ids(runtime["detector"]),
        ),
    })
    result = worker.snapshot()
    state_data = result["state_data"]
    api_health = get_api_health_monitor().snapshot()
    health_columns = st.columns(5)
    custom_detector = getattr(runtime["detector"], "is_custom_model", False)
    detector_validated = getattr(runtime["detector"], "is_validated", False)
    health_columns[0].badge(
        "YOLO: validated" if runtime["detector"].alerts_enabled else (
            "YOLO: review required" if custom_detector and detector_validated else (
                "YOLO: unvalidated" if custom_detector else "YOLO: generic"
            )
        ),
        color="green" if runtime["detector"].alerts_enabled else "orange",
    )
    lstm_available = runtime["lstm_model"] is not None
    health_columns[1].badge(
        "LSTM: ready" if lstm_available else "LSTM: unavailable",
        color="green" if lstm_available else "red",
    )
    pose_estimator = worker.pose_estimator
    pose_available = (
        pose_estimator is not None and pose_estimator.available
    )
    health_columns[2].badge(
        "MediaPipe: ready" if pose_available else (
            "MediaPipe: initializing"
            if pose_estimator is None and worker.is_running
            else "MediaPipe: unavailable"
        ),
        color="green" if pose_available else (
            "orange"
            if pose_estimator is None and worker.is_running
            else "red"
        ),
    )
    health_columns[3].badge(
        "ByteTrack: running" if worker.is_running else "ByteTrack: stopped",
        color="green" if worker.is_running else "gray",
    )
    health_columns[4].badge(
        "API: online" if api_health["healthy"] is True else (
            "API: offline" if api_health["healthy"] is False else "API: checking"
        ),
        color="green" if api_health["healthy"] is True else (
            "red" if api_health["healthy"] is False else "gray"
        ),
    )
    if pose_estimator is None:
        st.info("MediaPipe is initializing in the inference worker.")
    elif not pose_available:
        st.warning(pose_estimator.diagnostic)

    live_frame = None
    live_age_ms = None
    if capture.is_webcam:
        live_result = capture.read(0, count_drops=False)
        if live_result is not None:
            live_frame, sequence, captured_at, _ = live_result
            st.session_state["smartvision_live_sequence"] = sequence
            live_age_ms = max(0.0, (time.perf_counter() - captured_at) * 1000)
            if live_age_ms > 500:
                live_frame = None
    else:
        live_frame = result["last_processed_frame"]

    if live_frame is not None:
        st.image(
            live_frame,
            channels="BGR",
            width="stretch",
            alt="Latest webcam frame without delayed detection overlays",
        )
    elif capture.is_webcam:
        st.warning("Waiting for the camera to publish its first frame.")
    else:
        st.info("Waiting for the first processed video frame.")

    if capture.error and capture.error != "Video finished.":
        st.error(capture.error)
    elif capture.error == "Video finished.":
        st.info("Video finished.")

    if capture.is_webcam and live_age_ms is not None and live_age_ms > 500:
        st.warning(
            f"No fresh webcam frame has arrived for {live_age_ms:.0f} ms. "
            "The stale frame is not displayed."
        )
    if result["error"]:
        st.error(f"Inference failed: {result['error']}")

    metric_columns = st.columns(6)
    metric_columns[0].metric("Capture FPS", f"{capture.capture_fps:.1f}")
    metric_columns[1].metric("Inference FPS", f"{result['inference_fps']:.1f}")
    metric_columns[2].metric(
        "Inference frame latency",
        f"{result['frame_latency_ms']:.0f} ms",
    )
    metric_columns[3].metric(
        "Latest live-frame age",
        f"{live_age_ms:.0f} ms" if live_age_ms is not None else "Unavailable",
    )
    metric_columns[4].metric(
        "Total processing time",
        f"{result['processing_ms']:.0f} ms",
    )
    metric_columns[5].metric("Dropped frames", capture.dropped_frames)

    if capture.is_webcam and capture.buffer_setting_supported is False:
        st.caption(
            "Camera backend rejected the buffer-size hint; the capture worker "
            "still drains frames continuously."
        )
    st.caption(
        "The live panel always shows the newest captured frame. Inference uses "
        "a single latest-frame slot and may intentionally skip frames."
    )
    stage_timings = result["timings_ms"]
    if stage_timings:
        with st.container(border=True):
            st.markdown("**Measured inference stages (rolling mean / p95)**")
            timing_summary = result["timing_summary_ms"]
            timing_items = [
                (stage, timing_summary[stage])
                for stage in (
                    "yolo",
                    "mediapipe",
                    "lstm",
                    "bytetrack",
                    "postprocess",
                    "preprocess",
                    "alert_enqueue",
                    "total_processing",
                )
                if stage in timing_summary
            ]
            for offset in range(0, len(timing_items), 4):
                stage_columns = st.columns(
                    min(4, len(timing_items) - offset)
                )
                for column, (stage, summary) in zip(
                    stage_columns,
                    timing_items[offset:offset + 4],
                ):
                    column.metric(
                        stage.replace("_", " ").title(),
                        f"{summary['mean']:.1f} ms",
                        delta=f"p95 {summary['p95']:.1f} ms",
                    )
            delivery_health = runtime["threat_manager"].get_delivery_health()
            api_value = delivery_health["last_latency_ms"]
            api_columns = st.columns(3)
            api_columns[0].metric(
                "Latest API request",
                f"{api_value:.1f} ms" if api_value is not None else "Not measured",
            )
            prepare_value = delivery_health["prepare_ms"]
            api_columns[1].metric(
                "Latest alert preparation",
                f"{prepare_value:.1f} ms"
                if prepare_value is not None
                else "Not measured",
            )
            api_columns[2].metric(
                "Alerts queued",
                delivery_health["queued"],
            )

    st.markdown("**Object detections**")
    objects = state_data.get("object_detections", [])
    people = state_data.get("person_detections", [])
    if objects or people:
        st.caption(f"Inference timestamp: {result['last_processed_time']}")
        for detection in objects:
            confidence = detection.get("confidence")
            description = detection.get("class", "Unknown")
            if isinstance(confidence, (int, float)):
                description += f" · {confidence:.1%}"
            if not detection.get("is_project_class", False):
                description += " · detector-only class (not project-validated)"
            track_id = detection.get("track_id")
            if track_id is not None:
                description += f" · Track ID {track_id}"
            st.write(description)
        for person in people:
            confidence = person.get("confidence")
            description = person.get("class", "Person")
            if isinstance(confidence, (int, float)):
                description += f" · {confidence:.1%}"
            track_id = person.get("track_id")
            if track_id is not None:
                description += f" · Track ID {track_id}"
            st.write(description)
    else:
        st.caption("No supported objects detected in the latest inference.")

    st.markdown("**Action-recognition alerts**")
    action_detections = state_data.get("action_detections", [])
    if action_detections:
        for action in action_detections:
            st.write(
                f"{action['class']} · {action['confidence']:.1%} · "
                f"Track ID {action['track_id']}"
            )
    elif runtime["lstm_model"] is None:
        st.warning("Action recognition is unavailable: no valid LSTM checkpoint.")
    else:
        st.caption("No non-normal action was classified in the latest sequence.")

    delivery = runtime["threat_manager"].get_delivery_health()
    if delivery["last_latency_ms"] is not None:
        st.caption(
            "Latest API delivery: "
            f"{delivery['last_latency_ms']:.0f} ms · "
            f"{'healthy' if delivery['healthy'] else delivery['last_error']}"
        )
    st.caption(f"Alerts waiting to send: {delivery['queued']}")


@st.fragment(run_every="1s")
def render_inference_snapshot():
    worker = st.session_state.get("smartvision_worker")
    if worker is None:
        return
    result = worker.snapshot()
    st.subheader("Latest annotated inference")
    if result["last_processed_frame"] is None:
        st.caption("No inference result is available yet.")
        return
    snapshot_age_ms = max(
        0.0,
        (time.perf_counter() - result["last_processed_at"]) * 1000,
    )
    st.caption(
        f"Last completed inference · {snapshot_age_ms:.0f} ms ago. "
        "The live panel above is fresher; this frame shows the actual "
        "detection overlays for the completed inference."
    )
    st.image(
        result["last_processed_frame"],
        channels="BGR",
        width="stretch",
        alt="Latest completed inference with detection boxes and labels",
    )


@st.fragment(run_every="2s")
def render_incident_history(runtime):
    with st.expander("Incident history and alert delivery"):
        incidents = runtime["threat_manager"].get_incident_history(limit=5)
        if not incidents:
            st.caption("No incidents have been raised in this session.")
        for incident in incidents:
            st.write(
                f"{incident['created_at']} · {incident['event_type']} · "
                f"{incident['label']} · {incident['confidence']:.1%} · "
                f"{incident['status']}"
            )
            if incident["snapshot"]:
                st.image(
                    incident["snapshot"],
                    caption="Incident snapshot",
                    width=320,
                    alt="Incident snapshot",
                )
            if incident["error"]:
                st.caption(incident["error"])


# --------------------------------------------------
# LIVE DASHBOARD PAGE
# --------------------------------------------------

def run_streamlit_inference_page():
    st.title("SmartVision Pro")
    st.caption("Live detection and threat monitoring")
    st.sidebar.title("Detection source")

    runtime = init_models_and_helpers()

    if runtime is None or runtime.get("detector") is None:
        st.error(
            "YOLO model could not be loaded. "
            "Check your model configuration and file paths."
        )
        return

    detector = runtime["detector"]
    if detector.alerts_enabled:
        st.sidebar.success("Validated custom detector loaded; alerts enabled.")
    else:
        st.sidebar.warning(detector.model_diagnostic)

    available_ids = _available_class_ids(detector)
    safety_class_labels = [
        label
        for class_id, label in (
            (5, "person"),
            (2, "firearm"),
            (4, "knife/blade"),
        )
        if class_id in available_ids
    ]
    st.sidebar.caption(
        "Supported safety classes: "
        + (", ".join(safety_class_labels) if safety_class_labels else "none")
    )

    if st.sidebar.button("🔄 Reload Model & Settings"):
        if not stop_active_source():
            st.error(
                "The active video worker did not stop cleanly. "
                "The model was not reloaded to avoid duplicate workers."
            )
            return
        old_runtime = init_models_and_helpers()
        if old_runtime is not None:
            old_runtime["threat_manager"].close()
        init_models_and_helpers.clear()
        st.rerun()

    input_option = st.sidebar.selectbox(
        "Select Input Source",
        (
            "Live Webcam",
            "Upload Video",
            "Select from Input Folder",
        ),
    )

    source_path = None
    uploaded_temp_path = None

    if input_option == "Live Webcam":
        source_path = st.session_state.get(
            "smartvision_camera_index",
            WEBCAM_ID,
        )

        st.info(
            "Selected source: Live Webcam. "
            "Click Start Inference to begin."
        )

    elif input_option == "Select from Input Folder":
        if os.path.isdir(INPUT_VIDEO_DIR):
            video_files = sorted(
                filename
                for filename in os.listdir(INPUT_VIDEO_DIR)
                if filename.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                )
            )

            if video_files:
                selected_file = st.sidebar.selectbox(
                    "Choose a video",
                    video_files,
                )

                source_path = os.path.join(
                    INPUT_VIDEO_DIR,
                    selected_file,
                )

                st.info(f"Selected video: {selected_file}")
            else:
                st.warning(
                    f"No videos found in {INPUT_VIDEO_DIR}"
                )
        else:
            st.error(
                f"Input folder does not exist: {INPUT_VIDEO_DIR}"
            )

    elif input_option == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv"],
        )

        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]
            signature = (uploaded_file.name, uploaded_file.size)

            try:
                previous_path = st.session_state.get(
                    "smartvision_uploaded_temp_path"
                )
                previous_signature = st.session_state.get(
                    "smartvision_uploaded_signature"
                )
                if (
                    previous_path
                    and previous_signature == signature
                    and os.path.isfile(previous_path)
                ):
                    uploaded_temp_path = previous_path
                else:
                    if previous_path and os.path.isfile(previous_path):
                        os.remove(previous_path)
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        uploaded_temp_path = temp_file.name
                    st.session_state["smartvision_uploaded_signature"] = signature
                    st.session_state["smartvision_uploaded_temp_path"] = uploaded_temp_path

                source_path = uploaded_temp_path
                st.caption("Video upload is ready.")

            except OSError as exc:
                st.error(f"Could not save uploaded video: {exc}")

    start_col, stop_col = st.sidebar.columns(2)
    if start_col.button("Start", type="primary", key="start_smartvision"):
        if source_path is None:
            st.error("Select a valid video source before starting.")
        else:
            existing_capture = st.session_state.get("smartvision_capture")
            existing_worker = st.session_state.get("smartvision_worker")
            if (
                existing_capture is not None
                and existing_worker is not None
                and existing_capture.source == source_path
                and existing_capture.is_running
                and existing_worker.is_running
            ):
                st.info("This source is already running in this session.")
            elif not stop_active_source():
                st.error(
                    "The previous video worker did not stop cleanly. "
                    "The source was not reopened."
                )
            else:
                camera_lock = get_webcam_lock()
                webcam_lock_acquired = False
                if isinstance(source_path, int):
                    webcam_lock_acquired = camera_lock.acquire(
                        blocking=False
                    )
                    if not webcam_lock_acquired:
                        st.error(
                            "The webcam is already owned by another "
                            "SmartVision session in this server process."
                        )

                if isinstance(source_path, int) and not webcam_lock_acquired:
                    pass
                else:
                    if webcam_lock_acquired:
                        st.session_state["smartvision_webcam_lock_held"] = True

                    capture = LatestFrameCapture(source_path)
                    if capture.wait_until_ready():
                        st.session_state["smartvision_capture"] = capture
                        st.session_state["smartvision_source"] = source_path
                        source_name = (
                            f"camera_{capture.source}"
                            if capture.is_webcam
                            else os.path.basename(os.fspath(capture.source))
                        )
                        try:
                            worker = InferenceWorker(
                                capture,
                                runtime,
                                source_name,
                                settings={
                                    "confidence_threshold": st.session_state.get(
                                        "smartvision_confidence_threshold",
                                        CONFIDENCE_THRESHOLD,
                                    ),
                                    "image_size": st.session_state.get(
                                        "smartvision_image_size",
                                        YOLO_IMAGE_SIZE,
                                    ),
                                    "class_thresholds": st.session_state.get(
                                        "smartvision_class_thresholds",
                                        CLASS_CONFIDENCE_THRESHOLDS,
                                    ),
                                    "enabled_classes": st.session_state.get(
                                        "smartvision_enabled_classes",
                                        _available_class_ids(detector),
                                    ),
                                },
                            )
                        except Exception as exc:
                            if not stop_active_source():
                                st.error(
                                    "Worker initialization failed and capture "
                                    "shutdown is still pending."
                                )
                            st.error(f"Could not start inference: {exc}")
                        else:
                            st.session_state["smartvision_worker"] = worker
                            st.session_state["smartvision_live_sequence"] = 0
                            st.session_state.pop(
                                "smartvision_last_live_frame",
                                None,
                            )
                            st.session_state.pop(
                                "smartvision_last_live_at",
                                None,
                            )
                    else:
                        st.error(capture.error or "Could not start video capture.")
                        capture_stopped = capture.stop()
                        if not capture_stopped:
                            st.session_state["smartvision_capture"] = capture
                            st.error(
                                "Camera startup timed out and its capture "
                                "thread has not stopped; the camera remains "
                                "reserved until shutdown completes."
                            )
                        elif webcam_lock_acquired:
                            st.session_state.pop(
                                "smartvision_webcam_lock_held",
                                None,
                            )
                            camera_lock.release()

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
                        f"The temporary uploaded video could not be removed: {exc}"
                    )
            st.session_state.pop("smartvision_uploaded_signature", None)
        else:
            st.error(
                "The video worker did not stop cleanly. The source remains "
                "reserved; wait briefly and press Stop again."
            )

    capture = st.session_state.get("smartvision_capture")
    if capture is not None:
        st.subheader("Live detection")
        render_live_inference()
        render_inference_snapshot()
        render_incident_history(runtime)
    else:
        st.info("No active source. Choose a camera or video and select Start.")


@st.cache_data(ttl=10, max_entries=4, show_spinner=False)
def fetch_incidents(limit=100):
    try:
        response = requests.get(
            API_URL,
            params={"limit": limit},
            timeout=10,
        )
        if response.status_code != 200:
            return [], f"Incident API returned HTTP {response.status_code}."
        alerts = response.json()
        if isinstance(alerts, dict):
            for key in ("items", "incidents", "data"):
                if key in alerts:
                    alerts = alerts[key]
                    break
            else:
                return [], "Incident API returned an unexpected response format."
        if not isinstance(alerts, list):
            return [], "Incident API returned an unexpected response format."
        return alerts, None
    except requests.RequestException as exc:
        return [], f"Could not connect to the incident API: {exc}"
    except (ValueError, TypeError) as exc:
        return [], f"Could not read incident response: {exc}"


def backend_is_available():
    return get_api_health_monitor().snapshot()["healthy"] is True


def _render_status_header(runtime):
    capture = st.session_state.get("smartvision_capture")
    model_ok = runtime is not None and runtime.get("detector") is not None
    api_health = get_api_health_monitor().snapshot()
    api_ok = api_health["healthy"] is True
    camera_ok = capture is not None and capture.is_running
    camera_open = camera_ok and capture.opened
    header_columns = st.columns([2.2, 1, 1, 1, 1.1, 0.8])
    header_columns[0].markdown("## SmartVision Pro")
    header_columns[1].badge(
        "Camera connected" if camera_open else (
            "Camera recovering" if camera_ok else "Camera idle"
        ),
        color="green" if camera_open else (
            "orange" if camera_ok else "gray"
        ),
    )
    header_columns[2].badge(
        "AI model ready" if model_ok else "AI model unavailable",
        color="green" if model_ok else "red",
    )
    if api_health["healthy"] is None:
        api_label, api_color = "API checking", "gray"
    else:
        api_label = "API online" if api_ok else "API offline"
        api_color = "green" if api_ok else "red"
    header_columns[3].badge(api_label, color=api_color)
    header_columns[4].caption(datetime.now().strftime("%H:%M:%S"))
    light_theme = header_columns[5].toggle(
        "Light",
        key="smartvision_light_theme",
        help="Switch the dashboard appearance.",
    )
    if light_theme:
        st.html(
            """
            <style>
            :root {
                color-scheme: light;
                --background-color: #F3F6FB;
                --secondary-background-color: #E7EDF5;
                --text-color: #17263B;
                --primary-color: #2457D6;
            }
            [data-testid="stAppViewContainer"] {
                background-color: #F3F6FB !important;
            }
            [data-testid="stSidebar"] {
                background-color: #E7EDF5 !important;
            }
            [data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1px solid #D4DEEA;
                border-radius: 6px;
            }
            </style>
            """
        )


def _render_overview():
    st.title("Overview")
    alerts, alert_error = fetch_incidents()
    capture = st.session_state.get("smartvision_capture")
    worker = st.session_state.get("smartvision_worker")
    state = worker.snapshot() if worker is not None else {}
    active_cameras = int(
        capture is not None and capture.is_webcam and capture.is_running
    )
    latest_state = state.get("state_data", {})
    weapons = latest_state.get("weapon_detections", [])
    critical_alerts = sum(
        "CRITICAL" in str(alert.get("threat_level", "")).upper()
        for alert in alerts
    )
    fps = state.get("inference_fps", 0.0)

    with st.container(horizontal=True):
        st.metric("Active cameras", active_cameras, border=True)
        st.metric("People detected", latest_state.get("people_detected", 0), border=True)
        st.metric("Weapon detections", len(weapons), border=True)
        st.metric("Critical alerts (latest 100)", critical_alerts, border=True)
        st.metric("Inference FPS", f"{fps:.1f}", border=True)

    if alert_error:
        st.warning(alert_error)
    elif alerts:
        valid_records = [alert for alert in alerts if isinstance(alert, dict)]
        valid_alerts = pd.DataFrame(valid_records)
        if "created_at" in valid_alerts:
            valid_alerts["created_at"] = pd.to_datetime(
                valid_alerts["created_at"], errors="coerce", utc=True
            )
        left, right = st.columns(2)
        with left:
            st.subheader("Alerts by severity")
            if "threat_level" in valid_alerts:
                counts = valid_alerts["threat_level"].fillna("Unknown").value_counts()
                st.bar_chart(counts, horizontal=True)
        with right:
            st.subheader("Detections by class")
            if "label" in valid_alerts:
                counts = valid_alerts["label"].fillna("Unknown").value_counts()
                st.bar_chart(counts, horizontal=True)
        activity_columns = st.columns(2)
        with activity_columns[0]:
            st.subheader("Detections over time")
            if {"created_at", "label"}.issubset(valid_alerts.columns):
                timed_alerts = valid_alerts.dropna(subset=["created_at"])
                if not timed_alerts.empty:
                    hourly = timed_alerts.groupby(
                        pd.Grouper(key="created_at", freq="h")
                    ).size()
                    st.bar_chart(hourly)
                else:
                    st.caption("Stored incidents do not include valid timestamps.")
        with activity_columns[1]:
            st.subheader("Camera activity")
            if "camera_id" in valid_alerts:
                activity = valid_alerts["camera_id"].fillna("Unknown").value_counts()
                st.bar_chart(activity, horizontal=True)
    else:
        st.info("No stored incidents are available for analytics yet.")

    history = list(state.get("performance_history", []))
    if history:
        st.subheader("Measured inference performance")
        performance = pd.DataFrame(history).set_index("time")
        st.line_chart(performance, y=["fps", "latency_ms"])
    else:
        st.caption("Performance history will appear after inference starts.")


def _render_camera_management():
    st.title("Camera management")
    camera_index = st.number_input(
        "Webcam device index",
        min_value=0,
        max_value=16,
        value=int(st.session_state.get("smartvision_camera_index", WEBCAM_ID)),
        step=1,
        key="smartvision_camera_index",
    )
    capture = st.session_state.get("smartvision_capture")
    with st.container(border=True):
        st.markdown("**Configured webcam**")
        st.write(f"Device index: {camera_index}")
        if capture is not None and capture.is_running:
            st.success("Capture worker is active.")
            st.caption(f"Measured capture rate: {capture.capture_fps:.1f} FPS")
        else:
            st.info("No camera is currently streaming.")
    st.caption("Choose this device under Live Detection and select Start to connect.")


def _render_detection_settings(runtime):
    st.title("Detection settings")
    available_ids = _available_class_ids(runtime["detector"])
    supported_options = [
        (5, "Person"),
        (2, "Firearm"),
        (4, "Knife or blade"),
    ]
    supported_options = [item for item in supported_options if item[0] in available_ids]
    enabled = st.session_state.get(
        "smartvision_enabled_classes",
        available_ids,
    )
    enabled = set(enabled)
    if supported_options:
        for class_id, label in supported_options:
            enabled_value = st.checkbox(
                f"Enable {label.lower()} detection",
                value=class_id in enabled,
                key=f"enable_class_{class_id}",
            )
            if enabled_value and class_id in available_ids:
                enabled.add(class_id)
            else:
                enabled.discard(class_id)
        st.session_state["smartvision_enabled_classes"] = enabled
    else:
        st.warning("No supported safety classes are present in the loaded model.")

    st.session_state["smartvision_confidence_threshold"] = st.slider(
        "Minimum detection confidence",
        min_value=min(0.99, max(0.05, CONFIDENCE_THRESHOLD)),
        max_value=0.99,
        value=min(0.99, max(CONFIDENCE_THRESHOLD, float(st.session_state.get(
            "smartvision_confidence_threshold",
            CONFIDENCE_THRESHOLD,
        )))),
        step=0.05,
    )
    for class_id, label in supported_options:
        class_thresholds = st.session_state.get(
            "smartvision_class_thresholds",
            dict(CLASS_CONFIDENCE_THRESHOLDS),
        )
        class_thresholds = dict(class_thresholds)
        class_thresholds[class_id] = st.slider(
            f"{label} confidence threshold",
            min_value=0.05,
            max_value=0.99,
            value=min(0.99, max(0.05, float(class_thresholds.get(class_id, CONFIDENCE_THRESHOLD)))),
            step=0.05,
            key=f"confidence_class_{class_id}",
        )
        st.session_state["smartvision_class_thresholds"] = class_thresholds

    st.session_state["smartvision_image_size"] = st.select_slider(
        "YOLO input size",
        options=[320, 416, 512, 640, 768, 960, 1280],
        value=int(st.session_state.get("smartvision_image_size", YOLO_IMAGE_SIZE)),
    )
    runtime["threat_manager"].log_cooldown = st.slider(
        "Alert cooldown per source and event (seconds)",
        min_value=1,
        max_value=300,
        value=int(runtime["threat_manager"].log_cooldown),
        step=1,
    )
    st.caption(
        "Thresholds can only become stricter than the configured global minimum. "
        "Unavailable model classes cannot be enabled here."
    )


def _render_system_health(runtime):
    st.title("Model and system health")
    if runtime is None or runtime.get("detector") is None:
        st.error("The detector is unavailable.")
        return
    detector = runtime["detector"]
    device = runtime["device"]
    st.metric("Inference device", str(device))
    if detector.alerts_enabled:
        st.success("Validated custom YOLO weights are loaded; production alerts are enabled.")
    elif detector.is_custom_model and detector.is_validated:
        st.warning(
            "Six-class weights passed the recorded validation checks, but "
            "production alerts remain disabled until an operator reviews and "
            "approves the validation record."
        )
    elif detector.is_custom_model:
        st.warning(
            detector.model_diagnostic
        )
    else:
        st.warning(
            "A general COCO fallback model is loaded; custom safety weights "
            "are missing and production alerts are disabled."
        )
    st.write("Action recognition:", "Loaded" if runtime["lstm_model"] is not None else "Unavailable")
    if runtime["lstm_model"] is None:
        st.info("Fighting and shooting recognition are disabled until a valid LSTM checkpoint is installed.")
    worker = st.session_state.get("smartvision_worker")
    pose_estimator = worker.pose_estimator if worker is not None else None
    if pose_estimator is None:
        st.write("MediaPipe:", "Not initialized (start a stream to initialize it)")
    elif pose_estimator.available:
        st.success(f"MediaPipe: {pose_estimator.diagnostic}")
    else:
        st.warning(f"MediaPipe: {pose_estimator.diagnostic}")
    st.write(
        "ByteTrack:",
        "Running" if worker is not None and worker.is_running else "Not running",
    )
    api_online = backend_is_available()
    st.write("Backend API:", "Connected" if api_online else "Unavailable")
    if not api_online:
        st.warning(
            f"Could not reach {API_URL}. Check that the FastAPI service is "
            "running and SMARTVISION_API_BASE_URL points to it."
        )
    delivery = runtime["threat_manager"].get_delivery_health()
    st.write("Alert queue:", delivery["queued"], "pending")
    st.write(
        "Last alert delivery:",
        delivery["last_error"] or (
            "Delivered" if delivery["healthy"] else "No delivery attempted"
        ),
    )
    names = detector.class_names or {}
    st.write("Detector classes:", ", ".join(map(str, names.values() if isinstance(names, dict) else names)))


# --------------------------------------------------
# ALERT HISTORY
# --------------------------------------------------

def show_alert_history():
    st.title("Alert history")
    alerts, error = fetch_incidents(limit=200)
    if error:
        st.error(error)
        st.caption("Confirm the backend is running and the configured API endpoint is reachable.")
        return
    if not alerts:
        st.info("No alerts found.")
        return

    records = []
    invalid_rows = 0
    for alert in alerts:
        if not isinstance(alert, dict):
            invalid_rows += 1
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
        raw_created_at = row.get("created_at")
        row["created_at"] = (
            pd.to_datetime(
                str(raw_created_at),
                errors="coerce",
                utc=True,
            )
            if raw_created_at is not None
            else pd.NaT
        )
        row["threat_level"] = str(row.get("threat_level") or "Unknown")
        row["label"] = str(row.get("label") or "Unknown")
        row["camera_id"] = str(row.get("camera_id") or "Unknown")
        records.append(row)
    if invalid_rows:
        st.warning(f"Ignored {invalid_rows} malformed incident record(s).")
    if not records:
        st.info("The API returned no valid incident records.")
        return

    df = pd.DataFrame(records)
    search = st.text_input("Search incidents", placeholder="Class, camera, or severity")
    severity_values = sorted(df["threat_level"].dropna().unique().tolist())
    selected_severity = st.multiselect("Severity", severity_values, default=severity_values)
    class_values = sorted(df["label"].dropna().unique().tolist())
    selected_classes = st.multiselect("Class or event", class_values, default=class_values)
    date_values = df["created_at"].dropna()
    if not date_values.empty:
        min_date = date_values.min().date()
        max_date = date_values.max().date()
    else:
        min_date = max_date = date.today()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    filtered = df[
        df["threat_level"].isin(selected_severity)
        & df["label"].isin(selected_classes)
    ].copy()
    if search:
        searchable = filtered[
            ["threat_level", "label", "camera_id"]
        ].fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered[searchable.str.contains(search, case=False, na=False)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        start = pd.Timestamp(datetime.combine(start_date, datetime_time.min), tz="UTC")
        end = pd.Timestamp(datetime.combine(end_date, datetime_time.max), tz="UTC")
        filtered = filtered[
            filtered["created_at"].isna()
            | filtered["created_at"].between(start, end)
        ]

    table_columns = [
        column
        for column in (
            "created_at", "threat_level", "label", "confidence", "camera_id", "id"
        )
        if column in filtered.columns
    ]
    st.dataframe(filtered[table_columns], hide_index=True, width="stretch")
    st.download_button(
        "Export filtered alerts as CSV",
        data=filtered[table_columns].to_csv(index=False).encode("utf-8"),
        file_name="smartvision-alerts.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    st.subheader("Incident details")
    for _, alert in filtered.iterrows():
        title = (
            f"{alert['created_at'] if pd.notna(alert['created_at']) else 'Unknown time'} · "
            f"{alert['threat_level']} · {alert['label']}"
        )
        with st.expander(title):
            image_col, detail_col = st.columns([1, 2])
            with image_col:
                image_path = alert.get("image_path")
                if isinstance(image_path, str) and image_path:
                    candidate = os.path.normpath(
                        image_path if os.path.isabs(image_path)
                        else os.path.join(ROOT_DIR, image_path)
                    )
                    try:
                        within_root = os.path.commonpath(
                            [os.path.abspath(ROOT_DIR), os.path.abspath(candidate)]
                        ) == os.path.abspath(ROOT_DIR)
                    except ValueError:
                        within_root = False
                    if within_root and os.path.isfile(candidate):
                        st.image(candidate, caption="Incident snapshot", width="stretch", alt="Incident snapshot")
                    else:
                        st.caption("Snapshot is unavailable.")
                else:
                    st.caption("No snapshot was attached.")
            with detail_col:
                st.write("Incident ID:", alert.get("id") or "Unavailable")
                st.write("Camera:", alert.get("camera_id", "Unknown"))
                confidence = alert.get("confidence")
                st.write(
                    "Confidence:",
                    f"{confidence:.1%}" if isinstance(confidence, (float, int)) else "Unavailable",
                )
                incident_data = alert.get("data")
                if isinstance(incident_data, (dict, list)):
                    st.json(incident_data)
                elif incident_data is not None:
                    st.caption("Incident details are malformed.")


# --------------------------------------------------
# MAIN NAVIGATION
# --------------------------------------------------

def main():
    runtime = init_models_and_helpers()
    _render_status_header(runtime)
    app_mode = st.sidebar.radio(
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
    if app_mode == "Overview":
        _render_overview()
    elif app_mode == "Live detection":
        run_streamlit_inference_page()
    elif app_mode == "Alert history":
        show_alert_history()
    elif app_mode == "Camera management":
        _render_camera_management()
    elif app_mode == "Detection settings":
        if runtime is not None:
            _render_detection_settings(runtime)
    else:
        _render_system_health(runtime)


if __name__ == "__main__":
    main()