# pipeline/run_inference.py

import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

ROOT_DIR = str(Path(__file__).resolve().parent.parent)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------------------

from src.config import (
    YOLO_MODEL_PATH,
    LSTM_MODEL_PATH,
    INPUT_VIDEO_DIR,
    RESULTS_DIR,
    WEBCAM_ID,
    ALERTS_ENABLED,
    ALERT_ENDPOINT_VERIFIED,
    CONFIDENCE_THRESHOLD,
    YOLO_IMAGE_SIZE,
    YOLO_IOU_THRESHOLD,
    CLASS_CONFIDENCE_THRESHOLDS,
    ACTION_CONFIDENCE_THRESHOLD,
    SEQUENCE_LENGTH,
    PERSON_CLASS_ID,
    WEAPON_CLASS_IDS,
    GUN_CLASS_IDS,
    KNIFE_CLASS_ID,
    ATM_CLASS_ID,
    BACKPACK_CLASS_ID,
    HANDBAG_CLASS_ID,
    UNATTENDED_BAG_THRESHOLD,
    get_color,
    DRAW_PERSON_BOXES,
    DRAW_LABELS,
)

from src.modeling.detectors.yolo_detector import YOLODetector
from src.modeling.trackers.byte_tracker import ByteTracker
from src.modeling.pose_estimators.mediapipe import MediaPipeEstimator

from src.modeling.classifiers.lstm import (
    POSE_FEATURE_SIZE,
    ActionModelCompatibilityError,
    PoseSequenceBuffer,
    load_action_model,
    predict_action,
)

from src.core.threat_manager import ThreatManager
from src.utils.logger import setup_logger

logger = setup_logger()

# ---------------------------------------------------------------------
# PERFORMANCE SETTINGS
# ---------------------------------------------------------------------

# Set these environment variables before starting Python if needed.
# SMARTVISION_FRAME_WIDTH=640
# SMARTVISION_FRAME_HEIGHT=480
# SMARTVISION_DISPLAY_SCALE=0.75

FRAME_WIDTH = int(os.getenv("SMARTVISION_FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("SMARTVISION_FRAME_HEIGHT", "480"))
DISPLAY_SCALE = float(os.getenv("SMARTVISION_DISPLAY_SCALE", "1.0"))

FRAME_WIDTH = max(320, min(FRAME_WIDTH, 1920))
FRAME_HEIGHT = max(240, min(FRAME_HEIGHT, 1080))
DISPLAY_SCALE = max(0.25, min(DISPLAY_SCALE, 1.0))

# ---------------------------------------------------------------------
# DEMO MODE
# ---------------------------------------------------------------------

# DEMO ONLY: must match SMARTVISION_DEMO_MODE on the server.
# When ON, incidents are posted to the server without requiring a
# validated detector. The server labels every alert as unvalidated.
# Leave OFF (default) for any real use.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(ROOT_DIR) / ".env")
except ImportError:
    pass

DEMO_MODE = (
    os.getenv("SMARTVISION_DEMO_MODE", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------


def load_models():
    """Load YOLO, optional LSTM, device, and threat manager."""

    detector = YOLODetector()

    try:
        detector.load_model(YOLO_MODEL_PATH)
    except Exception:
        logger.exception("Unable to load the YOLO detector.")
        return None, None, None, None, None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Optional temporal action recognition.
    lstm_model = None
    inv_class_map = {}

    lstm_path = Path(LSTM_MODEL_PATH)

    if not lstm_path.is_absolute():
        lstm_path = Path(ROOT_DIR) / lstm_path

    if lstm_path.is_file():
        try:
            lstm_model, inv_class_map = load_action_model(
                lstm_path,
                device,
                sequence_length=SEQUENCE_LENGTH,
                feature_size=POSE_FEATURE_SIZE,
            )
            logger.info("LSTM action model loaded.")
        except (
            ActionModelCompatibilityError,
            OSError,
            RuntimeError,
            ValueError,
        ):
            logger.exception(
                "LSTM could not be loaded. "
                "Continuing with object detection only."
            )
            lstm_model = None
            inv_class_map = {}
        except Exception:
            logger.exception("Unexpected LSTM loading error.")
            lstm_model = None
            inv_class_map = {}
    else:
        logger.warning(
            "LSTM checkpoint not found: %s. "
            "Action recognition is disabled.",
            lstm_path,
        )

    if DEMO_MODE:
        # Demo only: skip the detector-validation requirement.
        production_alerts_enabled = True

        logger.warning(
            "DEMO MODE is ON: alerts are sent without detector "
            "validation. Do not use in production."
        )
    else:
        production_alerts_enabled = (
            detector.alerts_enabled
            and ALERTS_ENABLED
            and ALERT_ENDPOINT_VERIFIED
        )

        if not production_alerts_enabled:
            logger.warning(
                "Production alert delivery is disabled until the detector "
                "and alert endpoint are validated and the required settings "
                "are enabled."
            )

    threat_manager = ThreatManager(
        alerts_enabled=production_alerts_enabled
    )

    logger.info("YOLO initialized on %s.", device)

    return (
        detector,
        lstm_model,
        device,
        inv_class_map,
        threat_manager,
    )


# ---------------------------------------------------------------------
# CLASS MAPPING
# ---------------------------------------------------------------------

CANONICAL_NAME_MAP = {
    "person": PERSON_CLASS_ID,
    "backpack": BACKPACK_CLASS_ID,
    "handbag": HANDBAG_CLASS_ID,
    "atm": ATM_CLASS_ID,
    "gun": GUN_CLASS_IDS[0],
    "knife": KNIFE_CLASS_ID,
}


def get_canonical_class_id(
    detector: YOLODetector,
    raw_cls_id: int,
) -> int:
    """Map detector labels to the project's configured class IDs."""

    names = detector.class_names or {}

    if isinstance(names, dict):
        raw_name = names.get(raw_cls_id, "")
    elif isinstance(names, (list, tuple)):
        raw_name = (
            names[raw_cls_id]
            if 0 <= raw_cls_id < len(names)
            else ""
        )
    else:
        return -1

    name = str(raw_name).strip().lower()

    # Generic pretrained COCO labels are not proof of a
    # project-specific gun or weapon detection.
    if (
        name in ("atm", "gun", "knife")
        and not getattr(detector, "is_custom_model", False)
    ):
        return -1

    return CANONICAL_NAME_MAP.get(name, -1)


def _valid_box(box, shape=None):
    """Validate and optionally clip an XYXY bounding box."""

    try:
        coords = np.asarray(box, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if coords.shape != (4,) or not np.isfinite(coords).all():
        return None

    if shape is not None:
        height, width = shape[:2]
        coords[[0, 2]] = np.clip(coords[[0, 2]], 0, width - 1)
        coords[[1, 3]] = np.clip(coords[[1, 3]], 0, height - 1)

    x1, y1, x2, y2 = coords

    if x2 <= x1 or y2 <= y1:
        return None

    return [int(round(v)) for v in coords]


def _passes_confidence_threshold(
    detector,
    raw_class_id,
    confidence,
    base_threshold,
    class_thresholds=None,
    enabled_classes=None,
):
    class_id = get_canonical_class_id(detector, raw_class_id)

    if enabled_classes is not None and class_id not in enabled_classes:
        return False

    thresholds = dict(CLASS_CONFIDENCE_THRESHOLDS)

    if class_thresholds:
        thresholds.update(class_thresholds)

    threshold = max(
        float(base_threshold),
        float(thresholds.get(class_id, base_threshold)),
    )

    return confidence >= threshold


# ---------------------------------------------------------------------
# YOLO INFERENCE
# ---------------------------------------------------------------------


def run_yolo_detection(
    detector,
    frame,
    confidence_threshold=None,
    image_size=None,
):
    """Run YOLO using the project's configured thresholds."""

    threshold = (
        CONFIDENCE_THRESHOLD
        if confidence_threshold is None
        else max(CONFIDENCE_THRESHOLD, float(confidence_threshold))
    )

    return detector.predict(
        frame,
        conf=threshold,
        imgsz=YOLO_IMAGE_SIZE if image_size is None else image_size,
        iou=YOLO_IOU_THRESHOLD,
    )


def _extract_box_values(box):
    """Safely extract XYXY coordinates, confidence and class ID."""

    try:
        coordinates = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        raw_class = float(box.cls[0])

        if not np.isfinite(confidence) or not np.isfinite(raw_class):
            return None

        if not raw_class.is_integer():
            return None

        return coordinates, confidence, int(raw_class)

    except (TypeError, ValueError, IndexError, AttributeError):
        return None


def parse_detections(
    detector,
    results,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    class_thresholds=None,
    enabled_classes=None,
):
    """Extract person detections for ByteTrack."""

    people = []
    weapons = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            values = _extract_box_values(box)

            if values is None:
                continue

            raw_coords, confidence, raw_class_id = values
            coords = _valid_box(raw_coords)

            if coords is None:
                continue

            if not _passes_confidence_threshold(
                detector,
                raw_class_id,
                confidence,
                confidence_threshold,
                class_thresholds,
                enabled_classes,
            ):
                continue

            class_id = get_canonical_class_id(detector, raw_class_id)

            if class_id in WEAPON_CLASS_IDS:
                weapons.append(class_id)

            elif class_id == PERSON_CLASS_ID:
                people.append((coords, confidence, class_id))

    return people, weapons


def draw_weapon_boxes(
    detector,
    frame,
    results,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    class_thresholds=None,
    enabled_classes=None,
):
    """Draw objects and return structured detections."""

    visible_objects = []
    names = detector.class_names or {}

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            values = _extract_box_values(box)

            if values is None:
                continue

            raw_coords, confidence, raw_class_id = values
            coords = _valid_box(raw_coords, frame.shape)

            if coords is None:
                continue

            if not _passes_confidence_threshold(
                detector,
                raw_class_id,
                confidence,
                confidence_threshold,
                class_thresholds,
                enabled_classes,
            ):
                continue

            if isinstance(names, dict):
                class_name = str(names.get(raw_class_id, "Unknown"))
            elif isinstance(names, (list, tuple)):
                class_name = (
                    str(names[raw_class_id])
                    if 0 <= raw_class_id < len(names)
                    else "Unknown"
                )
            else:
                class_name = "Unknown"

            if class_name.strip().lower() == "person":
                continue

            class_id = get_canonical_class_id(detector, raw_class_id)
            is_weapon = class_id in WEAPON_CLASS_IDS

            detection = {
                "class": class_name,
                "class_id": class_id,
                "is_project_class": class_id >= 0,
                "is_weapon": is_weapon,
                "confidence": confidence,
                "bbox": coords,
            }

            visible_objects.append(detection)

            color = (0, 0, 255) if is_weapon else get_color(class_id)
            x1, y1, x2, y2 = coords

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if DRAW_LABELS:
                label = f"{class_name.upper()} {confidence:.0%}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

    return visible_objects


# ---------------------------------------------------------------------
# TRACK MANAGEMENT
# ---------------------------------------------------------------------


def _coerce_track_id(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if (
        not np.isfinite(number)
        or number < 0
        or not number.is_integer()
    ):
        return None

    return int(number)


def _maintain_track_history(
    tracked_objects,
    person_history,
    state_data,
    tracker,
):
    tracker_ids = getattr(tracked_objects, "tracker_id", None)

    active_ids = set()

    if tracker_ids is not None:
        active_ids = {
            track_id
            for value in tracker_ids
            if (track_id := _coerce_track_id(value)) is not None
        }

    missed = state_data.setdefault("missed_track_frames", {})

    tracker_impl = getattr(tracker, "tracker", None)

    retention = getattr(tracker, "max_time_lost", None)

    if retention is None:
        retention = getattr(
            tracker_impl,
            "max_time_lost",
            getattr(tracker_impl, "maximum_frames_without_update", 30),
        )

    retention = max(1, int(retention))

    for track_id in list(person_history):
        if track_id in active_ids:
            missed.pop(track_id, None)
            continue

        sequence = person_history[track_id]

        if hasattr(sequence, "clear"):
            sequence.clear()

        missed[track_id] = missed.get(track_id, 0) + 1

        if missed[track_id] > retention:
            person_history.pop(track_id, None)
            state_data.get("person_durations", {}).pop(track_id, None)
            missed.pop(track_id, None)


# ---------------------------------------------------------------------
# POSE AND ACTION RECOGNITION
# ---------------------------------------------------------------------


def update_person_history(
    pose_estimator,
    frame,
    tracked_objects,
    person_history,
    device,
    lstm_model,
    inv_class_map,
    threat_manager,
    context_detections,
    person_durations,
    is_unattended_bag,
    frame_id=0,
    source_name="Stream",
    stage_timings=None,
    action_detections=None,
    pose_frame=None,
):
    """Update person tracks, run optional LSTM, and record threats."""

    pose_source = frame if pose_frame is None else pose_frame
    image_height, image_width = pose_source.shape[:2]
    messages = []

    if len(tracked_objects) == 0:
        return messages

    visible_ids = [d["class_id"] for d in context_detections]
    has_atm = ATM_CLASS_ID in visible_ids

    for index in range(len(tracked_objects)):
        coords = _valid_box(tracked_objects.xyxy[index], pose_source.shape)

        if coords is None:
            continue

        track_id = _coerce_track_id(tracked_objects.tracker_id[index])

        if track_id is None:
            continue

        x1, y1, x2, y2 = coords

        confidence_values = getattr(tracked_objects, "confidence", None)
        confidence = (
            float(confidence_values[index])
            if confidence_values is not None
            else 0.0
        )

        if lstm_model is not None and track_id not in person_history:
            person_history[track_id] = PoseSequenceBuffer(
                sequence_length=SEQUENCE_LENGTH,
                feature_size=POSE_FEATURE_SIZE,
            )

        # Only process the person's crop rather than the whole frame.
        crop = pose_source[y1:y2, x1:x2]
        landmarks = None

        pose_start = time.perf_counter()

        if (
            lstm_model is not None
            and pose_estimator is not None
            and getattr(pose_estimator, "available", True)
            and crop.size > 0
        ):
            try:
                landmarks, _ = pose_estimator.predict(crop)
            except Exception:
                logger.exception("Pose inference failed for track %s", track_id)

        if stage_timings is not None:
            stage_timings["mediapipe"] = (
                stage_timings.get("mediapipe", 0.0)
                + (time.perf_counter() - pose_start) * 1000
            )

        sequence = person_history.get(track_id)

        if sequence is not None:
            sequence.append(landmarks)

        action_label = None
        action_confidence = 0.0

        if (
            lstm_model is not None
            and sequence is not None
            and sequence.ready
        ):
            lstm_start = time.perf_counter()

            try:
                prediction = predict_action(
                    lstm_model,
                    sequence,
                    inv_class_map,
                    device,
                    ACTION_CONFIDENCE_THRESHOLD,
                )
            except Exception:
                logger.exception("Action prediction failed for track %s", track_id)
                prediction = None

            if stage_timings is not None:
                stage_timings["lstm"] = (
                    stage_timings.get("lstm", 0.0)
                    + (time.perf_counter() - lstm_start) * 1000
                )

            if prediction is not None:
                action_label = str(prediction["class"])
                action_confidence = float(prediction["confidence"])

                if action_detections is not None:
                    action_detections.append({
                        "class": action_label,
                        "confidence": action_confidence,
                        "track_id": track_id,
                    })

        duration = person_durations.get(track_id, 0.0)

        threat_level, threat_color = threat_manager.determine_threat(
            action_label or "normal",
            visible_ids,
            duration,
            is_unattended_bag,
        )

        person_detection = {
            "class": "Person",
            "class_id": PERSON_CLASS_ID,
            "confidence": confidence,
            "bbox": coords,
            "track_id": track_id,
        }

        all_detections = [person_detection] + context_detections

        weapon_detections = [
            d for d in context_detections
            if d["class_id"] in WEAPON_CLASS_IDS
        ]

        event_label = action_label or "action recognition unavailable"
        event_confidence = action_confidence

        if action_label in (None, "normal") and weapon_detections:
            strongest = max(
                weapon_detections,
                key=lambda d: d["confidence"],
            )
            event_label = f"{strongest['class']} detection"
            event_confidence = strongest["confidence"]

        threat_manager.log_threat(
            track_id,
            source_name,
            event_label,
            event_confidence,
            threat_level,
            frame_id=frame_id,
            detections=all_detections,
            spatial_context={
                "atm_present": has_atm,
                "near_atm": has_atm,
            },
            frame=frame,
        )

        if DRAW_PERSON_BOXES:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if DRAW_LABELS:
                cv2.putText(
                    frame,
                    f"ID:{track_id} {int(duration)}s",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        if threat_level != "SAFE":
            messages.append(
                f"ID {track_id}: {threat_level} "
                f"({action_label or 'action unavailable'})"
            )

            if DRAW_LABELS:
                cv2.putText(
                    frame,
                    threat_level,
                    (x1, min(image_height - 5, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    threat_color,
                    2,
                )

    return messages


# ---------------------------------------------------------------------
# MAIN FRAME PIPELINE
# ---------------------------------------------------------------------


def process_frame(
    frame,
    detector,
    lstm_model,
    device,
    inv_class_map,
    threat_manager,
    tracker,
    pose_estimator,
    person_history,
    state_data=None,
    dt=0.05,
    source_name="Stream",
    confidence_threshold=None,
    image_size=None,
    class_thresholds=None,
    enabled_classes=None,
    timestamp=None,
):
    """Process a frame and update dashboard-ready detection statistics."""

    pipeline_start = time.perf_counter()

    if state_data is None:
        state_data = {}

    state_data.setdefault("person_durations", {})
    state_data.setdefault("unattended_bag_timer", 0.0)
    state_data.setdefault("frame_count", 0)
    state_data.setdefault("person_detections", [])
    state_data.setdefault("missed_track_frames", {})

    threshold = (
        CONFIDENCE_THRESHOLD
        if confidence_threshold is None
        else max(CONFIDENCE_THRESHOLD, float(confidence_threshold))
    )

    stage = {
        "mediapipe": 0.0,
        "lstm": 0.0,
    }
    action_detections = []

    # Avoid a full-frame copy when action recognition is unavailable.
    pose_frame = (
        frame.copy()
        if (
            lstm_model is not None
            and pose_estimator is not None
            and getattr(pose_estimator, "available", True)
        )
        else frame
    )

    begin_frame = getattr(threat_manager, "begin_frame", None)

    if begin_frame is not None:
        begin_frame()

    start = time.perf_counter()

    results = run_yolo_detection(
        detector,
        frame,
        threshold,
        image_size,
    )

    yolo_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()

    objects = draw_weapon_boxes(
        detector,
        frame,
        results,
        threshold,
        class_thresholds,
        enabled_classes,
    )

    visible_ids = [d["class_id"] for d in objects]

    has_bag = any(
        cid in (BACKPACK_CLASS_ID, HANDBAG_CLASS_ID)
        for cid in visible_ids
    )

    people, _ = parse_detections(
        detector,
        results,
        threshold,
        class_thresholds,
        enabled_classes,
    )

    postprocess_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()

    tracker.set_frame_rate(1.0 / max(float(dt), 0.001))

    if timestamp is None:
        tracked = tracker.update(people)
    else:
        tracked = tracker.update(people, timestamp=timestamp)

    stage["bytetrack"] = (time.perf_counter() - start) * 1000

    _maintain_track_history(
        tracked,
        person_history,
        state_data,
        tracker,
    )

    person_count = len(tracked)
    person_list = []

    tracker_ids = getattr(tracked, "tracker_id", None)
    confidences = getattr(tracked, "confidence", None)

    for index in range(person_count):
        track_id = (
            _coerce_track_id(tracker_ids[index])
            if tracker_ids is not None
            else None
        )

        confidence = (
            float(confidences[index])
            if confidences is not None
            else None
        )

        coords = _valid_box(tracked.xyxy[index], frame.shape)

        if coords is None:
            continue

        person_list.append({
            "class": "Person",
            "class_id": PERSON_CLASS_ID,
            "confidence": confidence,
            "bbox": coords,
            "track_id": track_id,
        })

        if track_id is not None:
            state_data["person_durations"][track_id] = (
                state_data["person_durations"].get(track_id, 0.0)
                + max(0.0, float(dt))
            )

    if has_bag and person_count == 0:
        state_data["unattended_bag_timer"] += max(0.0, float(dt))
    else:
        state_data["unattended_bag_timer"] = 0.0

    unattended = (
        state_data["unattended_bag_timer"]
        > UNATTENDED_BAG_THRESHOLD
    )

    start = time.perf_counter()

    messages = update_person_history(
        pose_estimator=pose_estimator,
        frame=frame,
        tracked_objects=tracked,
        person_history=person_history,
        device=device,
        lstm_model=lstm_model,
        inv_class_map=inv_class_map,
        threat_manager=threat_manager,
        context_detections=objects,
        person_durations=state_data["person_durations"],
        is_unattended_bag=unattended,
        frame_id=state_data["frame_count"],
        source_name=source_name,
        stage_timings=stage,
        action_detections=action_detections,
        pose_frame=pose_frame,
    )

    pose_action_ms = (time.perf_counter() - start) * 1000

    weapon_objects = [
        d for d in objects
        if d["class_id"] in WEAPON_CLASS_IDS
    ]

    # Log a weapon even when no person is present.
    if person_count == 0:
        for detection in weapon_objects:
            level, color = threat_manager.determine_threat(
                "normal",
                [detection["class_id"]],
            )

            if level == "SAFE":
                continue

            messages.append(
                f"Object detection: {detection['class']} "
                f"({detection['confidence']:.0%}); "
                "not action-confirmed"
            )

            threat_manager.log_threat(
                0,
                source_name,
                f"{detection['class']} detection",
                detection["confidence"],
                level,
                frame_id=state_data["frame_count"],
                detections=objects,
                spatial_context={
                    "atm_present": ATM_CLASS_ID in visible_ids,
                    "near_atm": False,
                },
                frame=frame,
            )

    if unattended and person_count == 0:
        level, color = threat_manager.determine_threat(
            "normal",
            visible_ids,
            0.0,
            True,
        )

        if level != "SAFE":
            messages.append(f"Unattended bag: {level}")

            threat_manager.log_threat(
                0,
                source_name,
                "unattended",
                1.0,
                level,
                frame_id=state_data["frame_count"],
                detections=objects,
                spatial_context={
                    "atm_present": ATM_CLASS_ID in visible_ids,
                    "near_atm": False,
                },
                frame=frame,
            )

            if DRAW_LABELS:
                cv2.putText(
                    frame,
                    f"UNATTENDED BAG: {level}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

    state_data["person_detections"] = person_list
    state_data["people_detected"] = len(person_list)
    state_data["object_detections"] = objects
    state_data["weapon_detections"] = weapon_objects
    state_data["action_detections"] = action_detections
    state_data["threat_messages"] = messages
    state_data["unattended_bag"] = unattended

    state_data["frame_count"] += 1

    state_data["timings_ms"] = {
        "yolo": round(yolo_ms, 2),
        "postprocess": round(postprocess_ms, 2),
        "bytetrack": round(stage["bytetrack"], 2),
        "mediapipe": round(stage["mediapipe"], 2),
        "lstm": round(stage["lstm"], 2),
        "track_pose_action": round(pose_action_ms, 2),
        "alert_enqueue": round(
            float(getattr(threat_manager, "frame_enqueue_ms", 0.0)),
            2,
        ),
        "total_pipeline": round(
            (time.perf_counter() - pipeline_start) * 1000,
            2,
        ),
    }

    return frame, messages


# ---------------------------------------------------------------------
# LOW-LATENCY CAMERA CAPTURE
# ---------------------------------------------------------------------


class LatestFrameCapture:
    """
    Read the camera on a background thread and retain only the newest frame.

    This prevents a slow detector from accumulating an ever-growing queue
    of old webcam frames. File-video playback should use VideoCapture
    directly so that no recorded frames are skipped.
    """

    def __init__(self, source):
        self.capture = (
            cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if os.name == "nt"
            else cv2.VideoCapture(source)
        )

        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.stopped = False
        self.thread = None

        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open webcam: {source}")

        # Some camera drivers ignore these settings; that is okay.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.thread = threading.Thread(
            target=self._reader,
            name="SmartVisionCameraReader",
            daemon=True,
        )
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            success, frame = self.capture.read()

            if not success:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame
                self.sequence += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None, self.sequence

            return True, self.frame.copy(), self.sequence

    def release(self):
        self.stopped = True

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        self.capture.release()


# ---------------------------------------------------------------------
# VIDEO / WEBCAM PIPELINE
# ---------------------------------------------------------------------


def run_offline_pipeline():
    """Run the detector on videos in the input directory or webcam."""

    print("Initializing SmartVision Pro...")

    (
        detector,
        lstm_model,
        device,
        inv_class_map,
        threat_manager,
    ) = load_models()

    if detector is None:
        logger.error("No usable YOLO model. Pipeline stopped.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    videos = []

    if os.path.isdir(INPUT_VIDEO_DIR):
        videos = [
            name for name in os.listdir(INPUT_VIDEO_DIR)
            if name.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
        ]

    sources = (
        [
            (
                os.path.join(INPUT_VIDEO_DIR, name),
                os.path.join(
                    RESULTS_DIR,
                    f"output_{Path(name).stem}.mp4",
                ),
            )
            for name in videos
        ]
        if videos
        else [
            (
                WEBCAM_ID,
                os.path.join(RESULTS_DIR, "webcam_session.mp4"),
            )
        ]
    )

    for source, output in sources:
        tracker = ByteTracker()
        pose_estimator = MediaPipeEstimator()
        person_history = {}

        try:
            _run_single_source_offline(
                source,
                output,
                detector,
                lstm_model,
                device,
                inv_class_map,
                threat_manager,
                tracker,
                pose_estimator,
                person_history,
            )
        finally:
            close_pose = getattr(pose_estimator, "close", None)

            if close_pose is not None:
                try:
                    close_pose()
                except Exception:
                    logger.exception("Could not close pose estimator.")


def _run_single_source_offline(
    source_path,
    output_path,
    detector,
    lstm_model,
    device,
    inv_class_map,
    threat_manager,
    tracker,
    pose_estimator,
    person_history,
):
    """Process and display one video or webcam source."""

    is_webcam = (
        isinstance(source_path, int)
        or (
            isinstance(source_path, str)
            and source_path.isdigit()
        )
    )

    source = int(source_path) if (
        isinstance(source_path, str) and source_path.isdigit()
    ) else source_path

    source_name = "Webcam" if is_webcam else f"File: {os.path.basename(source)}"

    logger.info("Starting source: %s", source_name)

    capture = None
    writer = None

    tracker.reset()
    person_history.clear()

    try:
        if is_webcam:
            capture = LatestFrameCapture(source)
            fps = 20.0
        else:
            capture = cv2.VideoCapture(source)

            if not capture.isOpened():
                logger.error("Cannot open video: %s", source)
                return

            fps = capture.get(cv2.CAP_PROP_FPS)

            if not np.isfinite(fps) or fps <= 0:
                fps = 20.0

        width = (
            int(capture.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            if is_webcam
            else int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        )
        height = (
            int(capture.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if is_webcam
            else int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        if width <= 0:
            width = FRAME_WIDTH

        if height <= 0:
            height = FRAME_HEIGHT

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            if not writer.isOpened():
                logger.warning("Could not create output video: %s", output_path)
                writer.release()
                writer = None

        state = {
            "person_durations": {},
            "unattended_bag_timer": 0.0,
            "frame_count": 0,
            "person_detections": [],
        }

        last_time = time.monotonic()
        last_sequence = -1
        last_display_time = last_time
        displayed_fps = 0.0

        while True:
            if is_webcam:
                success, frame, sequence_id = capture.read()

                # Do not run inference repeatedly on the same frame.
                if success and sequence_id == last_sequence:
                    time.sleep(0.002)
                    continue

                if success:
                    last_sequence = sequence_id
            else:
                success, frame = capture.read()

            if not success:
                break

            now = time.monotonic()

            dt = (
                max(0.001, now - last_time)
                if is_webcam
                else 1.0 / fps
            )
            last_time = now

            processed, messages = process_frame(
                frame=frame,
                detector=detector,
                lstm_model=lstm_model,
                device=device,
                inv_class_map=inv_class_map,
                threat_manager=threat_manager,
                tracker=tracker,
                pose_estimator=pose_estimator,
                person_history=person_history,
                state_data=state,
                dt=dt,
                source_name=source_name,
                timestamp=now,
            )

            # FPS shown is the actual processing/display rate.
            elapsed = max(now - last_display_time, 0.001)
            instantaneous_fps = 1.0 / elapsed
            displayed_fps = (
                instantaneous_fps
                if displayed_fps == 0.0
                else 0.8 * displayed_fps + 0.2 * instantaneous_fps
            )
            last_display_time = now

            cv2.putText(
                processed,
                f"FPS: {displayed_fps:.1f}",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            timings = state.get("timings_ms", {})

            cv2.putText(
                processed,
                f"People: {state.get('people_detected', 0)}",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                processed,
                f"YOLO: {timings.get('yolo', 0):.0f} ms",
                (15, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            if DEMO_MODE:
                cv2.putText(
                    processed,
                    "DEMO MODE - UNVALIDATED MODEL",
                    (15, 109),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )

            if DISPLAY_SCALE < 1.0:
                display_frame = cv2.resize(
                    processed,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                display_frame = processed

            if writer is not None:
                writer.write(processed)

            cv2.imshow("SmartVision Pro - Threat Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user.")

    except Exception:
        logger.exception("Pipeline failed for source: %s", source_name)

    finally:
        if capture is not None:
            capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    logger.info("Finished source: %s", source_name)


if __name__ == "__main__":
    run_offline_pipeline()