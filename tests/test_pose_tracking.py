import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pipeline.run_inference import update_person_history
from pipeline.run_inference import _run_single_source_offline
from pipeline.inference_worker import InferenceWorker
from src.modeling.classifiers.lstm import PoseSequenceBuffer
from src.modeling.pose_estimators.mediapipe import MediaPipeEstimator
from src.modeling.trackers.byte_tracker import ByteTracker


class ByteTrackIntegrationTests(unittest.TestCase):
    def test_track_id_survives_missing_frames_and_reset_starts_fresh_tracker(self):
        tracker = ByteTracker()
        first = tracker.update([([10, 10, 60, 80], 0.95, 5)])
        self.assertEqual(len(first), 1)
        initial_id = int(first.tracker_id[0])

        tracker.update([])
        tracker.update([])
        resumed = tracker.update([([12, 10, 62, 80], 0.94, 5)])
        self.assertEqual(len(resumed), 1)
        self.assertEqual(int(resumed.tracker_id[0]), initial_id)

        previous_tracker = tracker.tracker
        tracker.reset()
        self.assertIsNot(tracker.tracker, previous_tracker)
        restarted = tracker.update([([12, 10, 62, 80], 0.94, 5)])
        self.assertEqual(len(restarted), 1)

    def test_invalid_boxes_and_near_identical_duplicates_are_rejected(self):
        tracker = ByteTracker()
        filtered = tracker._validated_detections([
            ([10, 10, 50, 60], 0.91, 5),
            ([10.2, 10.2, 50.2, 60.2], 0.87, 5),
            ([20, 20, 20, 40], 0.99, 5),
            ([np.nan, 0, 20, 30], 0.99, 5),
            ([0, 0, 20, 30], np.inf, 5),
            ([0, 0, 20, 30], 0.99, 5.5),
        ])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][1], 0.91)


class MediaPipeFailureTests(unittest.TestCase):
    def test_worker_keeps_running_state_when_pose_constructor_raises(self):
        class IdleCapture:
            @property
            def is_running(self):
                return False

            def read(self, _last_sequence):
                return None

        with patch(
            "pipeline.inference_worker.MediaPipeEstimator",
            side_effect=RuntimeError("constructor failure"),
        ):
            worker = InferenceWorker(
                IdleCapture(),
                runtime={},
                source_name="pose-failure-test",
            )
            worker._thread.join(timeout=1.0)

        self.assertFalse(worker._thread.is_alive())
        self.assertIn("constructor failure", worker.pose_error)
        self.assertIsNone(worker.snapshot()["error"])
        self.assertTrue(worker.stop(timeout=0.1))

    def test_missing_task_model_disables_pose_without_raising(self):
        missing_model = Path(tempfile.gettempdir()) / "missing-pose-model.task"
        with patch(
            "src.modeling.pose_estimators.mediapipe.POSE_MODEL_PATH",
            missing_model,
        ):
            estimator = MediaPipeEstimator()

        self.assertFalse(estimator.available)
        self.assertIn("model is missing", estimator.diagnostic)
        landmarks, pose = estimator.predict(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertIsNone(landmarks)
        self.assertIsNone(pose)

    def test_tasks_inference_exception_returns_no_pose_and_diagnostic(self):
        estimator = MediaPipeEstimator.__new__(MediaPipeEstimator)
        estimator.available = True
        estimator.diagnostic = "ready"
        estimator.last_elapsed_ms = 0.0
        estimator.pose = SimpleNamespace(
            detect=Mock(side_effect=RuntimeError("test pose failure"))
        )
        estimator._task_api = True
        estimator._mp = SimpleNamespace(
            Image=Mock(return_value=object()),
            ImageFormat=SimpleNamespace(SRGB="srgb"),
        )

        landmarks, pose = estimator.predict(np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertIsNone(landmarks)
        self.assertIsNone(pose)
        self.assertIn("test pose failure", estimator.diagnostic)

    def test_each_pose_crop_is_stored_under_its_tracker_id(self):
        class Tracks:
            xyxy = np.asarray(
                [[0, 0, 8, 8], [12, 12, 20, 20]],
                dtype=np.float32,
            )
            tracker_id = np.asarray([101, 202], dtype=int)
            confidence = np.asarray([0.9, 0.9], dtype=np.float32)

            def __len__(self):
                return 2

        class PoseByCrop:
            available = True

            def predict(self, crop):
                value = float(crop[0, 0, 0])
                return [value] * 132, None

        class SafeThreatManager:
            def determine_threat(self, *_args):
                return "SAFE", (0, 255, 0)

            def log_threat(self, *_args, **_kwargs):
                return None

        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        frame[0:8, 0:8] = 30
        frame[12:20, 12:20] = 200
        person_history = {}

        update_person_history(
            pose_estimator=PoseByCrop(),
            frame=frame,
            tracked_objects=Tracks(),
            person_history=person_history,
            device="cpu",
            lstm_model=object(),
            inv_class_map={},
            threat_manager=SafeThreatManager(),
            context_detections=[],
            person_durations={101: 1.0, 202: 1.0},
            is_unattended_bag=False,
            pose_frame=frame.copy(),
        )

        self.assertEqual(set(person_history), {101, 202})
        self.assertEqual(person_history[101]._frames[0][0], 30.0)
        self.assertEqual(person_history[202]._frames[0][0], 200.0)

    def test_offline_source_change_resets_tracker_and_person_history(self):
        class ClosedCapture:
            released = False

            def isOpened(self):
                return False

            def release(self):
                self.released = True

        tracker = ByteTracker()
        previous_tracker = tracker.tracker
        person_history = {91: PoseSequenceBuffer(30)}
        capture = ClosedCapture()
        with patch(
            "pipeline.run_inference.cv2.VideoCapture",
            return_value=capture,
        ):
            _run_single_source_offline(
                0,
                None,
                detector=None,
                lstm_model=None,
                device="cpu",
                inv_class_map={},
                threat_manager=None,
                tracker=tracker,
                pose_estimator=None,
                person_history=person_history,
            )

        self.assertIsNot(tracker.tracker, previous_tracker)
        self.assertEqual(person_history, {})
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()