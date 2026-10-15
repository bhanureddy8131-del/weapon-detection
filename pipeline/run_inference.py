# pipeline/run_inference.py

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------
# PROJECT PATH
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
# MODEL LOADING
# ---------------------------------------------------------------------

def load_models():
    """
    Load the YOLO detector and optional LSTM classifier.

    Returns:
        detector, lstm_model, device, inv_class_map, threat_manager

    The YOLODetector class handles fallback to the root yolo11s.pt file.
    """

    detector = YOLODetector()

    try:
        detector.load_model(YOLO_MODEL_PATH)
    except Exception:
        logger.exception("Unable to load a YOLO model.")
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

    # LSTM is optional.
    lstm_model = None
    inv_class_map = {}

    lstm_path = Path(LSTM_MODEL_PATH)

    if not lstm_path.is_absolute():
        lstm_path = Path(ROOT_DIR) / lstm_path

    if not lstm_path.is_file():
        logger.warning(
            "LSTM checkpoint is missing: %s. "
            "Action recognition is disabled.",
            lstm_path,
        )
    else:
        try:
            lstm_model, inv_class_map = load_action_model(
                lstm_path,
                device,
                sequence_length=SEQUENCE_LENGTH,
                feature_size=POSE_FEATURE_SIZE,
            )
            logger.info("LSTM model loaded successfully.")
        except (ActionModelCompatibilityError, OSError, RuntimeError, ValueError) as exc:
            logger.error(
                "LSTM checkpoint is incompatible or unreadable; "
                "action recognition is disabled: %s",
                exc,
            )
            lstm_model = None
            inv_class_map = {}
        except Exception:
            logger.exception(
                "LSTM loading failed. Continuing without action recognition."
            )
            lstm_model = None
            inv_class_map = {}

    production_alerts_enabled = (
        detector.alerts_enabled
        and ALERTS_ENABLED
        and ALERT_ENDPOINT_VERIFIED
    )
    if detector.alerts_enabled and not production_alerts_enabled:
        logger.warning(
            "Production alert delivery remains disabled. Set both "
            "SMARTVISION_ALERTS_ENABLED and "
            "SMARTVISION_ALERT_ENDPOINT_VERIFIED only after verifying the "
            "configured FastAPI endpoint."
        )
    threat_manager = ThreatManager(alerts_enabled=detector.alerts_enabled)

    logger.info("YOLO initialized. Device: %s", device)

    return (
        detector,
        lstm_model,
        device,
        inv_class_map,
        threat_manager,
    )


# ---------------------------------------------------------------------
# DETECTION HELPERS
# ---------------------------------------------------------------------

def run_yolo_detection(
    detector: YOLODetector,
    frame: np.ndarray,
    confidence_threshold=None,
    image_size=None,
):
    """Run object detection on a frame."""
    return detector.predict(
        frame,
        conf=(
            CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else max(CONFIDENCE_THRESHOLD, confidence_threshold)
        ),
        imgsz=YOLO_IMAGE_SIZE if image_size is None else image_size,
        iou=YOLO_IOU_THRESHOLD,
    )


def _passes_confidence_threshold(
    detector,
    raw_class_id,
    confidence: float,
    base_threshold: float,
    class_thresholds: Optional[Dict[int, float]] = None,
    enabled_classes=None,
):
    canonical_class_id = get_canonical_class_id(
        detector,
        raw_class_id,
    )
    if enabled_classes is not None and canonical_class_id not in enabled_classes:
        return False
    thresholds = dict(CLASS_CONFIDENCE_THRESHOLDS)
    if class_thresholds:
        thresholds.update(class_thresholds)
    threshold = max(
        base_threshold,
        thresholds.get(canonical_class_id, base_threshold),
    )
    return confidence >= threshold


def _validated_box_coordinates(raw_coordinates, image_shape=None):
    try:
        coordinates = np.asarray(raw_coordinates, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        return None

    if image_shape is not None:
        image_height, image_width = image_shape[:2]
        coordinates[[0, 2]] = np.clip(coordinates[[0, 2]], 0, image_width - 1)
        coordinates[[1, 3]] = np.clip(coordinates[[1, 3]], 0, image_height - 1)

    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(value)) for value in coordinates]


# Only map labels that genuinely correspond to the project's
# weapon/context categories. Do not classify every object as a weapon.
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
    """Map a detector class name to the project's canonical class ID."""

    names = detector.class_names

    if not names:
        return raw_cls_id

    if isinstance(names, dict):
        raw_name = names.get(raw_cls_id, "")
    elif isinstance(names, (list, tuple)):
        raw_name = (
            names[raw_cls_id]
            if 0 <= raw_cls_id < len(names)
            else ""
        )
    else:
        return raw_cls_id

    canonical_name = str(raw_name).strip().lower()

    # COCO's generic "knife" is not the project's trained weapon class.
    if (
        canonical_name in ("atm", "gun", "knife")
        and not getattr(detector, "is_custom_model", False)
    ):
        return -1

    return CANONICAL_NAME_MAP.get(
        canonical_name,
        -1,
    )


def parse_detections(
    detector: YOLODetector,
    results,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    class_thresholds=None,
    enabled_classes=None,
) -> Tuple[List, List[int]]:
    """Extract person tracking detections and recognized weapon IDs."""

    detections_for_tracking = []
    visible_weapons = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            coordinates = _validated_box_coordinates(box.xyxy[0].tolist())
            if coordinates is None:
                continue
            try:
                x1, y1, x2, y2 = coordinates
                confidence = float(box.conf[0])
                raw_class_value = float(box.cls[0])
                if not np.isfinite(confidence) or not np.isfinite(raw_class_value):
                    continue
                if not raw_class_value.is_integer():
                    continue
                raw_class_id = int(raw_class_value)
            except (TypeError, ValueError, IndexError):
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

            class_id = get_canonical_class_id(
                detector,
                raw_class_id,
            )

            if class_id in WEAPON_CLASS_IDS:
                visible_weapons.append(class_id)

            elif class_id == PERSON_CLASS_ID:
                detections_for_tracking.append(
                    ([x1, y1, x2, y2], confidence, class_id)
                )

    return detections_for_tracking, visible_weapons


def draw_weapon_boxes(
    detector: YOLODetector,
    frame: np.ndarray,
    results,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    class_thresholds=None,
    enabled_classes=None,
) -> List[Dict]:
    """
    Draw detected objects.

    Only objects identified as weapons by their detector class are
    treated as weapons. Ordinary objects are not converted into knives.
    """

    visible_objects = []
    names = detector.class_names or {}

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            coordinates = _validated_box_coordinates(
                box.xyxy[0].tolist(),
                frame.shape,
            )
            if coordinates is None:
                continue
            try:
                x1, y1, x2, y2 = coordinates
                raw_class_value = float(box.cls[0])
                confidence = float(box.conf[0])
                if not np.isfinite(confidence) or not np.isfinite(raw_class_value):
                    continue
                if not raw_class_value.is_integer():
                    continue
                raw_class_id = int(raw_class_value)
            except (TypeError, ValueError, IndexError):
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
                class_name = str(
                    names.get(raw_class_id, "Unknown")
                )
            elif isinstance(names, (list, tuple)):
                class_name = (
                    str(names[raw_class_id])
                    if 0 <= raw_class_id < len(names)
                    else "Unknown"
                )
            else:
                class_name = "Unknown"

            if class_name.lower() == "person":
                continue

            class_id = get_canonical_class_id(
                detector,
                raw_class_id,
            )

            is_weapon = class_id in WEAPON_CLASS_IDS

            visible_objects.append({
                "class": class_name,
                "class_id": class_id,
                "is_project_class": class_id >= 0,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            })

            color = (
                (0, 0, 255)
                if is_weapon
                else get_color(class_id)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = f"{class_name.upper()} ({confidence:.2f})"

            label_y = max(20, y1 - 8)

            cv2.putText(
                frame,
                label,
                (max(0, x1), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    return visible_objects


# ---------------------------------------------------------------------
# PERSON TRACKING, POSE AND ACTION RECOGNITION
# ---------------------------------------------------------------------

def update_person_history(
    pose_estimator: MediaPipeEstimator,
    frame: np.ndarray,
    tracked_objects,
    person_history: Dict[int, PoseSequenceBuffer],
    device,
    lstm_model,
    inv_class_map,
    threat_manager: ThreatManager,
    context_detections: List[Dict],
    person_durations: Dict[int, float],
    is_unattended_bag: bool,
    frame_id: int = 0,
    source_name: str = "Stream",
    stage_timings: Optional[Dict[str, float]] = None,
    action_detections: Optional[List[Dict]] = None,
    pose_frame: Optional[np.ndarray] = None,
):
    """Process tracked people, optional action recognition, and alerts."""

    pose_source = frame if pose_frame is None else pose_frame
    image_height, image_width = pose_source.shape[:2]
    threat_messages = []

    if len(tracked_objects) == 0:
        return threat_messages

    visible_object_ids = [
        detection["class_id"]
        for detection in context_detections
    ]

    has_atm = ATM_CLASS_ID in visible_object_ids

    for index in range(len(tracked_objects)):
        x1, y1, x2, y2 = map(
            int,
            tracked_objects.xyxy[index],
        )

        track_id_value = _coerce_track_id(tracked_objects.tracker_id[index])
        if track_id_value is None:
            continue
        track_id = track_id_value

        confidence_values = getattr(
            tracked_objects,
            "confidence",
            None,
        )

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

        x1c = max(0, min(image_width, x1))
        y1c = max(0, min(image_height, y1))
        x2c = max(0, min(image_width, x2))
        y2c = max(0, min(image_height, y2))

        person_crop = pose_source[y1c:y2c, x1c:x2c]

        landmarks_list = None

        pose_started = time.perf_counter()
        if (
            lstm_model is not None
            and pose_estimator is not None
            and getattr(pose_estimator, "available", True)
            and person_crop.size > 0
        ):
            try:
                landmarks_list, _ = pose_estimator.predict(person_crop)
            except Exception:
                logger.exception(
                    "Pose inference failed for person track %s; "
                    "the action sequence will be reset.",
                    track_id,
                )
        if stage_timings is not None:
            stage_timings["mediapipe"] = (
                stage_timings.get("mediapipe", 0.0)
                + (time.perf_counter() - pose_started) * 1000
            )

        sequence = person_history.get(track_id)
        if sequence is not None:
            sequence.append(landmarks_list)

        action_label = None
        action_probability = 0.0

        if lstm_model is not None and sequence is not None and sequence.ready:
            lstm_started = time.perf_counter()
            prediction = predict_action(
                lstm_model,
            sequence,
                inv_class_map,
                device,
                ACTION_CONFIDENCE_THRESHOLD,
            )
            if stage_timings is not None:
                stage_timings["lstm"] = (
                    stage_timings.get("lstm", 0.0)
                    + (time.perf_counter() - lstm_started) * 1000
                )

            if prediction is not None:
                action_label = str(prediction["class"])
                action_probability = float(prediction["confidence"])
                if action_detections is not None:
                    action_detections.append({
                        "class": action_label,
                        "confidence": action_probability,
                        "track_id": track_id,
                    })

        duration = person_durations.get(track_id, 0.0)

        rule_action_label = action_label or "normal"
        threat_level, box_color = threat_manager.determine_threat(
            rule_action_label,
            visible_object_ids,
            duration,
            is_unattended_bag,
        )

        person_detection = {
            "class": "Person",
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "track_id": track_id,
        }

        all_detections = (
            [person_detection] + context_detections
        )

        spatial_context = {
            "atm_present": has_atm,
            "near_atm": has_atm,
        }

        weapon_detections = [
            detection
            for detection in context_detections
            if detection["class_id"] in WEAPON_CLASS_IDS
        ]
        event_label = action_label or "action recognition unavailable"
        event_confidence = action_probability
        if action_label in (None, "normal") and weapon_detections:
            strongest_weapon = max(
                weapon_detections,
                key=lambda detection: detection["confidence"],
            )
            event_label = f"{strongest_weapon['class']} detection"
            event_confidence = strongest_weapon["confidence"]

        threat_manager.log_threat(
            track_id,
            source_name,
            event_label,
            event_confidence,
            threat_level,
            frame_id=frame_id,
            detections=all_detections,
            spatial_context=spatial_context,
            frame=frame,
        )

        if DRAW_PERSON_BOXES:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            if DRAW_LABELS:
                label = f"ID:{track_id} ({int(duration)}s)"

                cv2.putText(
                    frame,
                    label,
                    (max(0, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        if threat_level != "SAFE":
            message = (
                f"ID {track_id}: {threat_level} "
                f"({action_label or 'action unavailable'})"
            )
            threat_messages.append(message)

            if DRAW_LABELS:
                cv2.putText(
                    frame,
                    threat_level,
                    (max(0, x1), min(image_height - 5, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2,
                )

    return threat_messages


def _coerce_track_id(value):
    try:
        numeric_id = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_id) or numeric_id < 0 or not numeric_id.is_integer():
        return None
    return int(numeric_id)


def _maintain_track_history(
    tracked_objects,
    person_history,
    state_data,
    tracker,
):
    tracker_ids = getattr(tracked_objects, "tracker_id", None)
    active_track_ids = set()
    if tracker_ids is not None:
        active_track_ids = {
            track_id
            for value in tracker_ids
            if (track_id := _coerce_track_id(value)) is not None
        }

    missed_frames = state_data.setdefault("missed_track_frames", {})
    tracker_impl = getattr(tracker, "tracker", None)
    retention_frames = max(
        1,
        int(getattr(tracker_impl, "max_time_lost", 1)),
    )

    for track_id in list(person_history):
        if track_id in active_track_ids:
            missed_frames.pop(track_id, None)
            continue

        sequence = person_history[track_id]
        if hasattr(sequence, "clear"):
            sequence.clear()
        missed_frames[track_id] = missed_frames.get(track_id, 0) + 1
        if missed_frames[track_id] > retention_frames:
            person_history.pop(track_id, None)
            state_data.get("person_durations", {}).pop(track_id, None)
            missed_frames.pop(track_id, None)


# ---------------------------------------------------------------------
# FRAME PROCESSING
# ---------------------------------------------------------------------

def process_frame(
    frame: np.ndarray,
    detector,
    lstm_model,
    device,
    inv_class_map,
    threat_manager: ThreatManager,
    tracker: ByteTracker,
    pose_estimator: MediaPipeEstimator,
    person_history: Dict[int, PoseSequenceBuffer],
    state_data: Optional[Dict] = None,
    dt: float = 0.05,
    source_name: str = "Stream",
    confidence_threshold: Optional[float] = None,
    image_size: Optional[int] = None,
    class_thresholds: Optional[Dict[int, float]] = None,
    enabled_classes=None,
):
    """
    Process one video frame.

    Returns:
        processed_frame, threat_messages
    """

    frame_started = time.perf_counter()
    if state_data is None:
        state_data = {
            "person_durations": {},
            "unattended_bag_timer": 0.0,
            "frame_count": 0,
            "person_detections": [],
        }

    state_data.setdefault("person_durations", {})
    state_data.setdefault("unattended_bag_timer", 0.0)
    state_data.setdefault("frame_count", 0)
    state_data.setdefault("person_detections", [])
    state_data.setdefault("missed_track_frames", {})
    base_confidence_threshold = (
        CONFIDENCE_THRESHOLD
        if confidence_threshold is None
        else max(CONFIDENCE_THRESHOLD, confidence_threshold)
    )

    stage_timings = {
        "mediapipe": 0.0,
        "lstm": 0.0,
    }
    action_detections = []
    pose_input_frame = (
        frame.copy()
        if lstm_model is not None
        and pose_estimator is not None
        and getattr(pose_estimator, "available", True)
        else frame
    )
    begin_alert_frame = getattr(threat_manager, "begin_frame", None)
    if begin_alert_frame is not None:
        begin_alert_frame()

    stage_started = time.perf_counter()
    results = run_yolo_detection(
        detector,
        frame,
        confidence_threshold,
        image_size,
    )
    yolo_ms = (time.perf_counter() - stage_started) * 1000

    stage_started = time.perf_counter()
    context_detections = draw_weapon_boxes(
        detector,
        frame,
        results,
        base_confidence_threshold,
        class_thresholds,
        enabled_classes,
    )

    visible_object_ids = [
        detection["class_id"]
        for detection in context_detections
    ]

    has_bag = any(
        class_id in (BACKPACK_CLASS_ID, HANDBAG_CLASS_ID)
        for class_id in visible_object_ids
    )

    detections_for_tracking, _ = parse_detections(
        detector,
        results,
        base_confidence_threshold,
        class_thresholds,
        enabled_classes,
    )
    postprocess_ms = (time.perf_counter() - stage_started) * 1000

    stage_started = time.perf_counter()
    tracker.set_frame_rate(1.0 / max(dt, 0.001))
    tracked_objects: Any = tracker.update(
        detections_for_tracking
    )
    stage_timings["bytetrack"] = (time.perf_counter() - stage_started) * 1000

    _maintain_track_history(
        tracked_objects,
        person_history,
        state_data,
        tracker,
    )

    person_count = len(tracked_objects)

    if has_bag and person_count == 0:
        state_data["unattended_bag_timer"] += max(
            0.0,
            dt,
        )
    else:
        state_data["unattended_bag_timer"] = 0.0

    is_unattended_bag = (
        state_data["unattended_bag_timer"]
        > UNATTENDED_BAG_THRESHOLD
    )

    tracked_people = []
    tracker_ids = getattr(tracked_objects, "tracker_id", None)
    confidence_values = getattr(tracked_objects, "confidence", None)
    for index in range(person_count):
        track_id = int(
            tracker_ids[index]
        ) if tracker_ids is not None else None
        confidence = (
            float(confidence_values[index])
            if confidence_values is not None
            else None
        )
        tracked_people.append({
            "class": "Person",
            "confidence": confidence,
            "track_id": track_id,
        })

        if track_id is not None:
            state_data["person_durations"][track_id] = (
                state_data["person_durations"].get(track_id, 0.0)
                + max(0.0, dt)
            )

    state_data["person_detections"] = tracked_people
    stage_started = time.perf_counter()
    threat_messages = update_person_history(
        pose_estimator=pose_estimator,
        frame=frame,
        tracked_objects=tracked_objects,
        person_history=person_history,
        device=device,
        lstm_model=lstm_model,
        inv_class_map=inv_class_map,
        threat_manager=threat_manager,
        context_detections=context_detections,
        person_durations=state_data["person_durations"],
        is_unattended_bag=is_unattended_bag,
        frame_id=state_data["frame_count"],
        source_name=source_name,
        stage_timings=stage_timings,
        action_detections=action_detections,
        pose_frame=pose_input_frame,
    )
    pose_action_ms = (time.perf_counter() - stage_started) * 1000

    state_data["people_detected"] = person_count
    state_data["weapon_detections"] = [
        detection
        for detection in context_detections
        if detection["class_id"] in WEAPON_CLASS_IDS
    ]
    state_data["action_detections"] = action_detections
    state_data["object_detections"] = context_detections

    state_data["frame_count"] += 1

    if person_count == 0:
        for detection in state_data["weapon_detections"]:
            threat_level, _ = threat_manager.determine_threat(
                "normal",
                [detection["class_id"]],
            )
            if threat_level == "SAFE":
                continue

            threat_messages.append(
                f"Object detection: {detection['class']} "
                f"({detection['confidence']:.0%}); not action-confirmed"
            )
            threat_manager.log_threat(
                0,
                source_name,
                f"{detection['class']} detection",
                detection["confidence"],
                threat_level,
                frame_id=state_data["frame_count"],
                detections=context_detections,
                spatial_context={
                    "atm_present": ATM_CLASS_ID in visible_object_ids,
                    "near_atm": False,
                },
                frame=frame,
            )

    if is_unattended_bag and person_count == 0:
        spatial_context = {
            "atm_present": ATM_CLASS_ID in visible_object_ids,
            "near_atm": False,
        }

        threat_level, threat_color = (
            threat_manager.determine_threat(
                "normal",
                visible_object_ids,
                0.0,
                True,
            )
        )

        if threat_level != "SAFE":
            threat_messages.append(
                f"Unattended bag: {threat_level}"
            )

            threat_manager.log_threat(
                0,
                source_name,
                "unattended",
                1.0,
                threat_level,
                frame_id=state_data["frame_count"],
                detections=context_detections,
                spatial_context=spatial_context,
                frame=frame,
            )

            if DRAW_LABELS:
                cv2.putText(
                    frame,
                    threat_level,
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    threat_color,
                    2,
                )

    state_data["timings_ms"] = {
        "yolo": yolo_ms,
        "postprocess": postprocess_ms,
        "bytetrack": stage_timings["bytetrack"],
        "mediapipe": stage_timings["mediapipe"],
        "lstm": stage_timings["lstm"],
        "alert_enqueue": getattr(
                threat_manager,
                "frame_enqueue_ms",
                0.0,
        ),
        "track_pose_action": pose_action_ms,
        "total_pipeline": (time.perf_counter() - frame_started) * 1000,
    }

    return frame, threat_messages


# ---------------------------------------------------------------------
# OFFLINE VIDEO / WEBCAM PIPELINE
# ---------------------------------------------------------------------

def run_offline_pipeline():
    """Run detection on videos in the input folder or on the webcam."""

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

    tracker = ByteTracker()
    pose_estimator = MediaPipeEstimator()
    person_history = {}

    os.makedirs(RESULTS_DIR, exist_ok=True)

    input_videos = []

    if os.path.isdir(INPUT_VIDEO_DIR):
        input_videos = [
            filename
            for filename in os.listdir(INPUT_VIDEO_DIR)
            if filename.lower().endswith(
                (".mp4", ".avi", ".mkv", ".mov")
            )
        ]

    if input_videos:
        for video_file in input_videos:
            input_path = os.path.join(
                INPUT_VIDEO_DIR,
                video_file,
            )

            output_path = os.path.join(
                RESULTS_DIR,
                f"output_{Path(video_file).stem}.mp4",
            )

            _run_single_source_offline(
                input_path,
                output_path,
                detector,
                lstm_model,
                device,
                inv_class_map,
                threat_manager,
                tracker,
                pose_estimator,
                person_history,
            )
    else:
        logger.info(
            "No input videos found. Starting webcam %s.",
            WEBCAM_ID,
        )

        _run_single_source_offline(
            WEBCAM_ID,
            os.path.join(RESULTS_DIR, "webcam_session.mp4"),
            detector,
            lstm_model,
            device,
            inv_class_map,
            threat_manager,
            tracker,
            pose_estimator,
            person_history,
        )


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
    """Read a video or webcam, process frames, and save output."""

    tracker.reset()
    person_history.clear()

    is_webcam = source_path == WEBCAM_ID

    source_name = (
        "Webcam"
        if is_webcam
        else f"File: {os.path.basename(source_path)}"
    )

    logger.info("Starting source: %s", source_name)

    capture = cv2.VideoCapture(source_path)

    if not capture.isOpened():
        logger.error("Cannot open source: %s", source_name)
        capture.release()
        return

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps or np.isnan(fps) or fps <= 0:
        fps = 20.0

    writer = None

    if output_path and width > 0 and height > 0:
        os.makedirs(
            os.path.dirname(output_path) or ".",
            exist_ok=True,
        )

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter.fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            logger.warning(
                "Could not create output video: %s",
                output_path,
            )
            writer.release()
            writer = None

    state_data = {
        "person_durations": {},
        "unattended_bag_timer": 0.0,
        "frame_count": 0,
    }

    last_time = time.time()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            current_time = time.time()

            if is_webcam:
                dt = max(
                    0.001,
                    current_time - last_time,
                )
            else:
                dt = 1.0 / fps

            last_time = current_time

            processed_frame, _ = process_frame(
                frame=frame,
                detector=detector,
                lstm_model=lstm_model,
                device=device,
                inv_class_map=inv_class_map,
                threat_manager=threat_manager,
                tracker=tracker,
                pose_estimator=pose_estimator,
                person_history=person_history,
                state_data=state_data,
                dt=dt,
            )

            if writer is not None:
                writer.write(processed_frame)

            cv2.imshow(
                "SmartVision Pro - Threat Detection",
                processed_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    logger.info("Finished source: %s", source_name)


if __name__ == "__main__":
    run_offline_pipeline()
