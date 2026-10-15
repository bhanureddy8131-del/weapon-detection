import json
import hashlib
import queue
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import requests

from pipeline.inference_worker import InferenceWorker, prepare_frame
from pipeline.latest_frame_capture import LatestFrameCapture
from pipeline.run_inference import draw_weapon_boxes, get_canonical_class_id
from scripts.prepare_detection_dataset import (
    _convert_label_row,
    prepare_detection_dataset,
)
from src.core.threat_manager import ThreatManager
from src.modeling.detectors.yolo_detector import YOLODetector
from src.modeling.trackers.byte_tracker import ByteTracker


class _Response:
    status_code = 201

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _StreamingCapture:
    def __init__(self, frames_before_disconnect=None):
        self.frames_before_disconnect = frames_before_disconnect
        self.frames_read = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def get(self, *_):
        return 30.0

    def read(self):
        if self.released:
            return False, None
        if (
            self.frames_before_disconnect is not None
            and self.frames_read >= self.frames_before_disconnect
        ):
            return False, None
        time.sleep(0.002)
        self.frames_read += 1
        return True, np.full(
            (8, 8, 3),
            self.frames_read % 255,
            dtype=np.uint8,
        )

    def release(self):
        self.released = True


class _FakeDetector(YOLODetector):
    def __init__(self, is_custom_model):
        self.class_names = {
            0: "person",
            39: "bottle",
            43: "knife",
        }
        self.is_custom_model = is_custom_model


class LivePipelineTests(unittest.TestCase):
    def test_webcam_capture_keeps_only_the_latest_frame_and_stops(self):
        fake_capture = _StreamingCapture()
        with patch(
            "pipeline.latest_frame_capture.cv2.VideoCapture",
            return_value=fake_capture,
        ):
            capture = LatestFrameCapture(0)
            self.assertTrue(capture.wait_until_ready(timeout=1.0))
            time.sleep(0.03)
            latest = capture.read(0)

            self.assertIsNotNone(latest)
            frame, sequence, captured_at, skipped = latest
            self.assertGreater(sequence, 1)
            self.assertEqual(int(frame[0, 0, 0]), sequence % 255)
            self.assertGreaterEqual(skipped, 0)
            self.assertIsNotNone(captured_at)
            self.assertGreater(capture.capture_fps, 0)
            self.assertTrue(capture.stop(timeout=1.0))
            self.assertTrue(fake_capture.released)

    def test_webcam_capture_reconnects_after_consecutive_read_failures(self):
        captures = []

        def make_capture(*_):
            capture = _StreamingCapture(
                frames_before_disconnect=1 if not captures else None
            )
            captures.append(capture)
            return capture

        with patch(
            "pipeline.latest_frame_capture.cv2.VideoCapture",
            side_effect=make_capture,
        ):
            capture = LatestFrameCapture(0)
            self.assertTrue(capture.wait_until_ready(timeout=1.0))
            deadline = time.monotonic() + 3.0
            while (
                len(captures) < 2 or not capture.opened
            ) and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertGreaterEqual(len(captures), 2)
            self.assertTrue(capture.opened)
            self.assertIsNone(capture.error)
            self.assertIsNotNone(capture.read(0))
            self.assertTrue(capture.stop(timeout=1.0))
            self.assertTrue(all(item.released for item in captures))

    def test_worker_timing_summary_reports_mean_and_p95(self):
        summary = InferenceWorker._summarize_timings([
            {"yolo": 100.0, "bytetrack": 2.0},
            {"yolo": 200.0, "bytetrack": 4.0},
            {"yolo": 300.0, "bytetrack": 6.0},
        ])
        self.assertEqual(summary["yolo"]["mean"], 200.0)
        self.assertAlmostEqual(summary["yolo"]["p95"], 290.0)
        self.assertEqual(summary["bytetrack"]["mean"], 4.0)

    def test_polygon_annotation_converts_to_axis_aligned_yolo_box(self):
        row, converted = _convert_label_row(
            "2 0.2 0.3 0.8 0.3 0.8 0.7 0.2 0.7",
            Path("sample.txt"),
            class_count=6,
        )
        self.assertTrue(converted)
        self.assertEqual(row, "2 0.50000000 0.50000000 0.60000000 0.40000000")

    def test_box_annotation_is_preserved_and_invalid_class_rejected(self):
        row, converted = _convert_label_row(
            "5 0.5 0.5 0.2 0.4",
            Path("sample.txt"),
            class_count=6,
        )
        self.assertFalse(converted)
        self.assertEqual(row, "5 0.50000000 0.50000000 0.20000000 0.40000000")
        with self.assertRaisesRegex(ValueError, "Class id 6"):
            _convert_label_row(
                "6 0.5 0.5 0.2 0.4",
                Path("sample.txt"),
                class_count=6,
            )

    def test_dataset_preparation_hardlinks_images_and_converts_only_staged_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "source"
            for split in ("train", "valid", "test"):
                image_dir = source_root / split / "images"
                label_dir = source_root / split / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                (image_dir / "sample.jpg").write_bytes(b"test image placeholder")
                (label_dir / "sample.txt").write_text(
                    "2 0.2 0.3 0.8 0.3 0.8 0.7 0.2 0.7\n"
                    "5 0.5 0.5 0.2 0.4\n",
                    encoding="utf-8",
                )

            source_yaml = root / "source.yaml"
            source_yaml.write_text(
                "path: " + str(source_root) + "\n"
                "train: train/images\n"
                "val: valid/images\n"
                "test: test/images\n"
                "nc: 6\n"
                "names:\n"
                "  0: atm\n"
                "  1: backpack\n"
                "  2: gun\n"
                "  3: handbag\n"
                "  4: knife\n"
                "  5: person\n",
                encoding="utf-8",
            )
            staged_root = root / "staged"
            staged_root.mkdir()
            staged_yaml, counts = prepare_detection_dataset(
                source_yaml,
                staged_root,
            )

            self.assertEqual(counts["linked_images"], 3)
            self.assertEqual(counts["converted_polygon_rows"], 3)
            self.assertEqual(
                (staged_root / "train" / "labels" / "sample.txt").read_text(
                    encoding="utf-8"
                ).splitlines()[0],
                "2 0.50000000 0.50000000 0.60000000 0.40000000",
            )
            self.assertIn(
                "0.2 0.4",
                (source_root / "train" / "labels" / "sample.txt").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertTrue(staged_yaml.is_file())

    def test_bytetrack_advances_on_empty_frames_and_keeps_track(self):
        tracker = ByteTracker()

        first = tracker.update([([10, 10, 100, 120], 0.95, 5)])
        self.assertEqual(len(first), 1)
        first_track_id = int(first.tracker_id[0])

        empty = tracker.update([])
        self.assertEqual(len(empty), 0)

        second = tracker.update([([11, 10, 101, 120], 0.95, 5)])
        self.assertEqual(len(second), 1)
        self.assertEqual(int(second.tracker_id[0]), first_track_id)

    def test_bytetrack_lost_window_tracks_inference_cadence(self):
        tracker = ByteTracker()
        tracker.set_frame_rate(2.5)
        self.assertEqual(tracker.tracker.max_time_lost, 5)
        first = tracker.update([([10, 10, 100, 120], 0.95, 5)])
        first_track_id = int(first.tracker_id[0])

        for _ in range(4):
            tracker.update([])

        resumed = tracker.update([([11, 10, 101, 120], 0.95, 5)])
        self.assertEqual(len(resumed), 1)
        self.assertEqual(int(resumed.tracker_id[0]), first_track_id)

    def test_generic_coco_knife_is_not_project_weapon(self):
        generic = _FakeDetector(is_custom_model=False)
        custom = _FakeDetector(is_custom_model=True)
        self.assertEqual(get_canonical_class_id(generic, 43), -1)
        self.assertEqual(get_canonical_class_id(custom, 43), 4)

        box = SimpleNamespace(
            xyxy=np.asarray([[10, 12, 40, 52]], dtype=np.float32),
            conf=np.asarray([0.91], dtype=np.float32),
            cls=np.asarray([43], dtype=np.float32),
        )
        detections = draw_weapon_boxes(
            generic,
            np.zeros((64, 64, 3), dtype=np.uint8),
            [SimpleNamespace(boxes=[box])],
            confidence_threshold=0.1,
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "knife")
        self.assertEqual(detections[0]["bbox"], [10, 12, 40, 52])
        self.assertFalse(detections[0]["is_project_class"])

    def test_preprocessing_preserves_640px_camera_frames(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIs(prepare_frame(frame), frame)
        resized = prepare_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertEqual(resized.shape, (360, 640, 3))

    def test_custom_detector_requires_operator_approval_for_alerts(self):
        from src.config import EXPECTED_DETECTOR_CLASSES

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            models_directory = root / "models"
            models_directory.mkdir()
            weights_path = models_directory / "custom.pt"
            weights_path.write_bytes(b"x" * 12000)
            digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            validation_path = weights_path.with_suffix(
                weights_path.suffix + ".validation.json"
            )
            record = {
                "validation_completed": True,
                "weights_sha256": digest,
                "classes": list(EXPECTED_DETECTOR_CLASSES),
                "validation_metrics": {"metrics/mAP50(B)": 0.8},
                "validation_summary": {
                    "precision": 0.8,
                    "recall": 0.7,
                    "map50": 0.8,
                    "map50_95": 0.6,
                },
                "representative_validation_completed": True,
                "production_alerts_approved": False,
            }
            validation_path.write_text(json.dumps(record), encoding="utf-8")
            fake_model = SimpleNamespace(
                names={
                    index: name
                    for index, name in enumerate(EXPECTED_DETECTOR_CLASSES)
                }
            )
            with (
                patch(
                    "src.modeling.detectors.yolo_detector.ROOT_DIR",
                    str(root),
                ),
                patch(
                    "src.modeling.detectors.yolo_detector.YOLO",
                    return_value=fake_model,
                ),
            ):
                detector = YOLODetector()
                detector.load_model(str(weights_path))
                self.assertTrue(detector.is_custom_model)
                self.assertTrue(detector.is_validated)
                self.assertFalse(detector.alerts_enabled)

                record["production_alerts_approved"] = True
                validation_path.write_text(
                    json.dumps(record),
                    encoding="utf-8",
                )
                approved_detector = YOLODetector()
                approved_detector.load_model(str(weights_path))
                self.assertTrue(approved_detector.alerts_enabled)

    def test_alert_delivery_reports_success_and_failure_and_deduplicates(self):
        session = _Session([
            _Response(),
            requests.ConnectionError("test backend unavailable"),
            _Response(),
        ])
        session_patcher = patch(
            "src.core.threat_manager.requests.Session",
            return_value=session,
        )
        session_patcher.start()
        self.addCleanup(session_patcher.stop)

        alerts_gate = patch("src.core.threat_manager.ALERTS_ENABLED", True)
        endpoint_gate = patch(
            "src.core.threat_manager.ALERT_ENDPOINT_VERIFIED",
            True,
        )
        alerts_gate.start()
        endpoint_gate.start()
        self.addCleanup(alerts_gate.stop)
        self.addCleanup(endpoint_gate.stop)

        manager = ThreatManager(alerts_enabled=True)
        self.addCleanup(manager.close)
        self.assertTrue(manager.alerts_enabled)
        manager.log_threat(
            7,
            "camera_test",
            "knife detection",
            0.91,
            "HIGH: WEAPON DETECTED",
        )
        manager.log_threat(
            7,
            "camera_test",
            "knife detection",
            0.91,
            "HIGH: WEAPON DETECTED",
        )
        manager.log_threat(
            8,
            "camera_test",
            "gun detection",
            0.96,
            "HIGH: WEAPON DETECTED",
        )

        deadline = time.monotonic() + 5
        while manager._alert_queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(manager._alert_queue.unfinished_tasks, 0)
        self.assertEqual(len(session.requests), 3)
        first_request = session.requests[0][1]
        self.assertEqual(first_request["timeout"], (0.5, 2.0))
        self.assertIn("_multipart", first_request["files"])
        self.assertNotIn("file", first_request["files"])
        posted_payload = json.loads(first_request["data"]["incident_data"])
        self.assertEqual(
            set(posted_payload["data"]),
            {
                "frame_id",
                "detections",
                "action_classification",
                "spatial_context",
            },
        )
        from server.schemas import IncidentCreate

        self.assertEqual(
            IncidentCreate(**posted_payload).camera_id,
            "camera_test",
        )
        incidents = manager.get_incident_history()
        self.assertEqual(incidents[0]["status"], "delivered (HTTP 201)")
        self.assertEqual(incidents[0]["attempts"], 2)
        self.assertEqual(
            incidents[1]["status"],
            "delivered (HTTP 201)",
        )
        self.assertTrue(manager.get_delivery_health()["healthy"])

    def test_failed_delivery_records_all_retries_and_never_reports_success(self):
        session = _Session([
            requests.ConnectionError("backend offline"),
            requests.ConnectionError("backend offline"),
            requests.ConnectionError("backend offline"),
        ])
        gates = [
            patch("src.core.threat_manager.ALERTS_ENABLED", True),
            patch("src.core.threat_manager.ALERT_ENDPOINT_VERIFIED", True),
            patch("src.core.threat_manager.ALERT_RETRY_BACKOFF_SECONDS", 0.0),
            patch("src.core.threat_manager.requests.Session", return_value=session),
        ]
        for gate in gates:
            gate.start()
            self.addCleanup(gate.stop)

        manager = ThreatManager(alerts_enabled=True)
        self.addCleanup(manager.close)
        manager.log_threat(
            17,
            "camera_test",
            "knife detection",
            0.91,
            "HIGH: WEAPON DETECTED",
        )
        deadline = time.monotonic() + 5
        while manager._alert_queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)

        incident = manager.get_incident_history(limit=1)[0]
        self.assertEqual(len(session.requests), 3)
        self.assertEqual(incident["attempts"], 3)
        self.assertEqual(incident["status"], "failed")
        self.assertIn("backend offline", incident["error"])
        self.assertFalse(manager.get_delivery_health()["healthy"])

    def test_production_gate_disabled_records_local_event_without_sending(self):
        with (
            patch("src.core.threat_manager.ALERTS_ENABLED", True),
            patch("src.core.threat_manager.ALERT_ENDPOINT_VERIFIED", False),
            patch("src.core.threat_manager.requests.Session") as session,
        ):
            manager = ThreatManager(alerts_enabled=True)
            self.addCleanup(manager.close)
            self.assertFalse(manager.alerts_enabled)
            self.assertIsNone(manager._alert_thread)
            manager.log_threat(
                17,
                "camera_test",
                "knife detection",
                0.91,
                "HIGH: WEAPON DETECTED",
            )
            manager.log_threat(
                17,
                "camera_test",
                "knife detection",
                0.91,
                "HIGH: WEAPON DETECTED",
            )

        session.assert_not_called()
        local_events = manager.get_incident_history()
        self.assertEqual(len(local_events), 1)
        self.assertEqual(
            local_events[0]["status"],
            "not sent: production alerts disabled",
        )
        self.assertIn("verified FastAPI endpoint", local_events[0]["error"])

    def test_full_alert_queue_is_reported_and_can_be_retried(self):
        manager = ThreatManager(alerts_enabled=False)
        manager.alerts_enabled = True
        manager._alert_queue = queue.Queue(maxsize=1)
        payload = {
            "camera_id": "camera_test",
            "threat_level": "HIGH: WEAPON DETECTED",
            "label": "KNIFE DETECTION",
            "confidence": 0.91,
            "data": {
                "event_type": "object_detection",
                "action_classification": {
                    "class": "knife detection",
                    "confidence": 0.91,
                    "track_id": 7,
                },
            },
        }
        with (
            patch("src.config.SERVER_URL", "http://test/incidents"),
            patch(
                "src.core.threat_manager.time.time",
                side_effect=[100.0, 102.0],
            ),
        ):
            self.assertIsNotNone(
                manager._send_alert_async("http://test/incidents", payload)
            )
            manager.log_threat(
                7,
                "camera_test",
                "knife detection",
                0.91,
                "HIGH: WEAPON DETECTED",
            )
            failed = manager.get_incident_history(limit=1)[0]
            self.assertEqual(failed["status"], "failed: alert queue is full")
            self.assertFalse(manager.threat_logs)

            manager._alert_queue.get_nowait()
            manager._alert_queue.task_done()
            manager.log_threat(
                7,
                "camera_test",
                "knife detection",
                0.91,
                "HIGH: WEAPON DETECTED",
            )

        self.assertIn(
            ("camera_test", 7, "object_detection", "knife detection"),
            manager.threat_logs,
        )
        self.assertEqual(manager.get_delivery_health()["queued"], 1)


if __name__ == "__main__":
    unittest.main()
