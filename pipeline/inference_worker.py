"""Run video inference off the Streamlit script thread."""

import copy
import logging
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from pipeline.run_inference import process_frame
from src.config import (
    CLASS_CONFIDENCE_THRESHOLDS,
    CONFIDENCE_THRESHOLD,
    YOLO_IMAGE_SIZE,
)
from src.modeling.pose_estimators.mediapipe import MediaPipeEstimator
from src.modeling.trackers.byte_tracker import ByteTracker

logger = logging.getLogger(__name__)

MAX_INFERENCE_WIDTH = 640
MAX_TEMPORAL_GAP_SECONDS = 1.0
TIMING_HISTORY_SIZE = 120


def annotate_source_frame(frame, object_detections, person_detections, inference_shape):
    """Draw inference-space detections on an original-resolution source frame."""
    source_height, source_width = frame.shape[:2]
    inference_height, inference_width = inference_shape[:2]
    scale_x = source_width / max(1, inference_width)
    scale_y = source_height / max(1, inference_height)
    scale = min(scale_x, scale_y)
    thickness = max(2, round(scale * 2))
    font_scale = max(0.55, min(1.0, 0.55 * scale))
    annotated = frame.copy()

    detections = [
        (detection, (40, 190, 240))
        for detection in object_detections
    ] + [
        (detection, (50, 210, 70))
        for detection in person_detections
    ]
    for detection, color in detections:
        box = detection.get("bbox")
        if box is None or len(box) != 4:
            continue
        x1 = int(round(box[0] * scale_x))
        y1 = int(round(box[1] * scale_y))
        x2 = int(round(box[2] * scale_x))
        y2 = int(round(box[3] * scale_y))
        x1 = min(max(x1, 0), source_width - 1)
        y1 = min(max(y1, 0), source_height - 1)
        x2 = min(max(x2, 0), source_width - 1)
        y2 = min(max(y2, 0), source_height - 1)
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(detection.get("class", "Object"))
        confidence = detection.get("confidence")
        if isinstance(confidence, (int, float)):
            label += f" {confidence:.0%}"
        track_id = detection.get("track_id")
        if track_id is not None and track_id >= 0:
            label += f"  ID {track_id}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_top = max(0, y1 - label_height - baseline - 8)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(source_width - 1, x1 + label_width + 8), y1),
            (24, 30, 36),
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(label_height, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def prepare_frame(frame: np.ndarray, max_width: int = MAX_INFERENCE_WIDTH):
    """Resize wide frames before CPU inference while preserving aspect ratio."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    resized_height = max(1, int(height * max_width / width))
    return cv2.resize(
        frame,
        (max_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


class InferenceWorker:
    """Consume only the newest webcam frame and publish thread-safe results."""

    def __init__(
        self,
        capture,
        runtime: Dict[str, Any],
        source_name: str,
        settings: Optional[Dict[str, Any]] = None,
    ):
        self.capture = capture
        self.runtime = runtime
        self.source_name = source_name
        self._stop_event = threading.Event()
        self._settings_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._settings = self._normalize_settings(settings or {})
        self._last_sequence = 0
        self._last_captured_at = None
        self._last_inference_at = None
        self._person_history = {}
        self._state_data = {
            "person_durations": {},
            "unattended_bag_timer": 0.0,
            "frame_count": 0,
        }
        self.tracker = ByteTracker()
        self.pose_estimator = None
        self.pose_error = None
        self._snapshot = {
            "last_processed_frame": None,
            "last_processed_sequence": 0,
            "last_captured_at": None,
            "last_processed_at": None,
            "processing_ms": 0.0,
            "frame_latency_ms": 0.0,
            "inference_fps": 0.0,
            "processed_frames": 0,
            "timings_ms": {},
            "timing_summary_ms": {},
            "timing_history": deque(maxlen=TIMING_HISTORY_SIZE),
            "capture_resolution": None,
            "inference_resolution": None,
            "recent_detections": deque(maxlen=100),
            "state_data": {},
            "threat_messages": [],
            "performance_history": deque(maxlen=300),
            "error": None,
        }
        self._thread = threading.Thread(
            target=self._run,
            name=f"smartvision-inference-{source_name}",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _normalize_settings(settings):
        return {
            "confidence_threshold": settings.get(
                "confidence_threshold",
                CONFIDENCE_THRESHOLD,
            ),
            "image_size": settings.get("image_size", YOLO_IMAGE_SIZE),
            "class_thresholds": dict(
                settings.get(
                    "class_thresholds",
                    CLASS_CONFIDENCE_THRESHOLDS,
                )
            ),
            "enabled_classes": set(settings.get("enabled_classes", ())),
        }

    @property
    def is_running(self):
        return self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def thread_id(self):
        return self._thread.ident

    def update_settings(self, settings):
        with self._settings_lock:
            self._settings = self._normalize_settings(settings)

    def snapshot(self):
        with self._snapshot_lock:
            values = dict(self._snapshot)
            values["timings_ms"] = dict(values["timings_ms"])
            values["state_data"] = copy.deepcopy(values["state_data"])
            values["threat_messages"] = list(values["threat_messages"])
            values["performance_history"] = list(
                values["performance_history"]
            )
            values["timing_history"] = list(values["timing_history"])
            values["timing_summary_ms"] = copy.deepcopy(
                values["timing_summary_ms"]
            )
            values["recent_detections"] = list(values["recent_detections"])
            return values

    def stop(self, timeout=2.0):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    @staticmethod
    def _summarize_timings(history):
        summary = {}
        stage_names = {
            stage
            for sample in history
            for stage in sample
        }
        for stage in stage_names:
            values = [
                sample[stage]
                for sample in history
                if stage in sample
            ]
            if values:
                summary[stage] = {
                    "mean": float(np.mean(values)),
                    "p95": float(np.percentile(values, 95)),
                }
        return summary

    def _run(self):
        try:
            self._run_loop()
        except Exception as exc:
            logger.exception(
                "Inference worker failed for %s.",
                self.source_name,
            )
            with self._snapshot_lock:
                self._snapshot["error"] = str(exc)
        finally:
            if self.pose_estimator is not None:
                try:
                    self.pose_estimator.close()
                except Exception as exc:
                    logger.exception(
                        "Could not close MediaPipe for %s.",
                        self.source_name,
                    )
                    with self._snapshot_lock:
                        previous_error = self._snapshot["error"]
                        self._snapshot["error"] = (
                            f"{previous_error}; " if previous_error else ""
                        ) + f"MediaPipe shutdown failed: {exc}"
            stop_capture = getattr(self.capture, "stop", None)
            if callable(stop_capture):
                try:
                    if not stop_capture():
                        logger.error(
                            "Capture source %s did not stop after its "
                            "inference worker exited.",
                            self.source_name,
                        )
                except Exception:
                    logger.exception(
                        "Could not stop capture source %s after its "
                        "inference worker exited.",
                        self.source_name,
                    )

    def _run_loop(self):
        if self.runtime.get("lstm_model") is not None:
            try:
                self.pose_estimator = MediaPipeEstimator()
            except Exception as exc:
                self.pose_estimator = None
                self.pose_error = str(exc)
                logger.exception(
                    "MediaPipe initialization failed for %s; continuing "
                    "without pose/action inference.",
                    self.source_name,
                )
        while not self._stop_event.is_set():
            result = self.capture.read(self._last_sequence)
            if result is None:
                if not self.capture.is_running:
                    break
                self._stop_event.wait(0.005)
                continue

            frame, sequence, captured_at, _ = result
            self._last_sequence = sequence
            if captured_at is None:
                continue
            source_frame = frame
            capture_resolution = (frame.shape[1], frame.shape[0])

            dt = 0.05
            if self._last_captured_at is not None:
                dt = max(0.001, captured_at - self._last_captured_at)
                if dt > MAX_TEMPORAL_GAP_SECONDS:
                    self._person_history.clear()
                    self._state_data["unattended_bag_timer"] = 0.0
                    self._state_data["person_durations"] = {}
                    self._state_data["missed_track_frames"] = {}
                    self.tracker.reset()
                dt = min(dt, 1.0)
            self._last_captured_at = captured_at

            inference_started = time.perf_counter()
            preprocess_started = time.perf_counter()
            frame = prepare_frame(frame)
            inference_resolution = (frame.shape[1], frame.shape[0])
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
            with self._settings_lock:
                settings = dict(self._settings)
                settings["class_thresholds"] = dict(
                    self._settings["class_thresholds"]
                )
                settings["enabled_classes"] = set(
                    self._settings["enabled_classes"]
                )

            try:
                with (
                    self.runtime["inference_lock"],
                    torch.inference_mode(),
                ):
                    processed_frame, threat_messages = process_frame(
                        frame,
                        self.runtime["detector"],
                        self.runtime["lstm_model"],
                        self.runtime["device"],
                        self.runtime["inv_class_map"],
                        self.runtime["threat_manager"],
                        self.tracker,
                        self.pose_estimator,
                        self._person_history,
                        state_data=self._state_data,
                        dt=dt,
                        timestamp=captured_at,
                        source_name=self.source_name,
                        confidence_threshold=settings[
                            "confidence_threshold"
                        ],
                        image_size=settings["image_size"],
                        class_thresholds=settings["class_thresholds"],
                        enabled_classes=settings["enabled_classes"],
                    )
            except Exception as exc:
                logger.exception(
                    "Inference failed for %s at frame %s.",
                    self.source_name,
                    sequence,
                )
                with self._snapshot_lock:
                    self._snapshot["error"] = str(exc)
                continue

            self._state_data["object_detections"] = list(
                self._state_data.get("object_detections", [])
            )
            overlay_started = time.perf_counter()
            if source_frame.shape[:2] != frame.shape[:2]:
                processed_frame = annotate_source_frame(
                    source_frame,
                    self._state_data.get("object_detections", []),
                    self._state_data.get("person_detections", []),
                    frame.shape,
                )
            overlay_ms = (time.perf_counter() - overlay_started) * 1000
            completed_at = time.perf_counter()
            processing_ms = (completed_at - inference_started) * 1000
            frame_latency_ms = max(0.0, (completed_at - captured_at) * 1000)
            stage_timings = dict(
                self._state_data.get("timings_ms", {})
            )
            stage_timings["preprocess"] = preprocess_ms
            stage_timings["render_overlay"] = overlay_ms
            stage_timings["total_processing"] = processing_ms
            if self._last_inference_at is None:
                inference_fps = 0.0
            else:
                instant_fps = 1.0 / max(
                    completed_at - self._last_inference_at,
                    1e-6,
                )
                previous_fps = self._snapshot["inference_fps"]
                inference_fps = (
                    instant_fps
                    if previous_fps == 0.0
                    else 0.8 * previous_fps + 0.2 * instant_fps
                )
            self._last_inference_at = completed_at

            detection_time = time.strftime("%Y-%m-%d %H:%M:%S")
            recent_detections = [
                {
                    "timestamp": detection_time,
                    "class": detection.get("class", "Unknown"),
                    "confidence": detection.get("confidence"),
                    "track_id": detection.get("track_id"),
                }
                for detection in (
                    self._state_data.get("object_detections", [])
                    + self._state_data.get("person_detections", [])
                )
            ]

            with self._snapshot_lock:
                history = self._snapshot["performance_history"]
                history.append({
                    "time": time.strftime("%H:%M:%S"),
                    "fps": inference_fps,
                    "latency_ms": frame_latency_ms,
                })
                timing_history = self._snapshot["timing_history"]
                timing_history.append(stage_timings)
                detection_history = self._snapshot["recent_detections"]
                detection_history.extend(recent_detections)
                self._snapshot.update({
                    "last_processed_frame": processed_frame,
                    "last_processed_sequence": sequence,
                    "last_captured_at": captured_at,
                    "last_processed_at": completed_at,
                    "last_processed_time": time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "processing_ms": processing_ms,
                    "frame_latency_ms": frame_latency_ms,
                    "inference_fps": inference_fps,
                    "processed_frames": (
                        self._snapshot["processed_frames"] + 1
                    ),
                    "timings_ms": stage_timings,
                    "timing_summary_ms": self._summarize_timings(
                        timing_history
                    ),
                    "capture_resolution": capture_resolution,
                    "inference_resolution": inference_resolution,
                    "recent_detections": detection_history,
                    "state_data": copy.deepcopy(self._state_data),
                    "threat_messages": list(threat_messages or []),
                    "performance_history": history,
                    "error": None,
                })
