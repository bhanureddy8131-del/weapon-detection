import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from src.modeling.classifiers.lstm import (
    ACTION_ARCHITECTURE,
    ACTION_CLASSES,
    POSE_FEATURE_LAYOUT,
    POSE_FEATURE_SIZE,
    ActionModelCompatibilityError,
    BidirectionalLSTM,
    PoseSequenceBuffer,
    format_action_prediction,
    load_action_model,
    load_action_model_from_checkpoint,
    predict_action,
)
from pipeline.run_inference import (
    _maintain_track_history,
    process_frame,
    update_person_history,
)


SEQUENCE_LENGTH = 30
CLASS_MAP = {"normal": 0, "fighting": 1, "shooting": 2}


def make_checkpoint(model=None):
    model = model or BidirectionalLSTM(
        POSE_FEATURE_SIZE,
        8,
        1,
        len(ACTION_CLASSES),
        dropout_rate=0.4,
    )
    return {
        "model_state_dict": model.state_dict(),
        "class_map": CLASS_MAP,
        "input_size": POSE_FEATURE_SIZE,
        "hidden_size": 8,
        "num_layers": 1,
        "bidirectional": True,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_layout": POSE_FEATURE_LAYOUT,
        "architecture": ACTION_ARCHITECTURE,
        "dropout_rate": 0.4,
    }


class PoseSequenceBufferTests(unittest.TestCase):
    def test_full_valid_sequence_has_expected_tensor_shape(self):
        sequence = PoseSequenceBuffer(SEQUENCE_LENGTH)
        for index in range(SEQUENCE_LENGTH - 1):
            self.assertTrue(sequence.append([float(index)] * POSE_FEATURE_SIZE))
            self.assertFalse(sequence.ready)

        self.assertTrue(sequence.append([1.0] * POSE_FEATURE_SIZE))
        tensor = sequence.as_tensor("cpu")
        self.assertTrue(sequence.ready)
        self.assertEqual(
            tuple(tensor.shape),
            (1, SEQUENCE_LENGTH, POSE_FEATURE_SIZE),
        )
        self.assertEqual(tensor.dtype, torch.float32)

    def test_missing_or_invalid_pose_clears_partial_sequence(self):
        sequence = PoseSequenceBuffer(SEQUENCE_LENGTH)
        for _ in range(SEQUENCE_LENGTH - 1):
            sequence.append([0.1] * POSE_FEATURE_SIZE)

        self.assertFalse(sequence.append(None))
        self.assertEqual(len(sequence), 0)
        with self.assertRaisesRegex(ValueError, "complete valid sequence"):
            sequence.as_tensor()

        self.assertFalse(sequence.append([0.1] * (POSE_FEATURE_SIZE - 1)))
        self.assertEqual(len(sequence), 0)
        self.assertFalse(sequence.append([float("nan")] * POSE_FEATURE_SIZE))
        self.assertEqual(len(sequence), 0)

    def test_missing_track_retains_id_but_resets_pose_sequence_until_expiry(self):
        sequence = PoseSequenceBuffer(SEQUENCE_LENGTH)
        for _ in range(SEQUENCE_LENGTH):
            sequence.append([0.1] * POSE_FEATURE_SIZE)
        person_history = {91: sequence}
        state_data = {"person_durations": {91: 4.0}}
        tracker = SimpleNamespace(
            tracker=SimpleNamespace(max_time_lost=2)
        )
        no_tracks = SimpleNamespace(tracker_id=np.asarray([], dtype=int))

        _maintain_track_history(
            no_tracks,
            person_history,
            state_data,
            tracker,
        )
        self.assertIn(91, person_history)
        self.assertFalse(person_history[91].ready)
        self.assertEqual(len(person_history[91]), 0)

        same_track_resumed = SimpleNamespace(
            tracker_id=np.asarray([91], dtype=int)
        )
        _maintain_track_history(
            same_track_resumed,
            person_history,
            state_data,
            tracker,
        )
        self.assertIn(91, person_history)
        self.assertNotIn(91, state_data["missed_track_frames"])
        self.assertFalse(person_history[91].ready)

        for _ in range(3):
            _maintain_track_history(
                no_tracks,
                person_history,
                state_data,
                tracker,
            )
        self.assertNotIn(91, person_history)
        self.assertNotIn(91, state_data["person_durations"])


class LSTMCheckpointTests(unittest.TestCase):
    def test_missing_checkpoint_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_action_model(
                    Path(directory) / "missing.pth",
                    "cpu",
                    SEQUENCE_LENGTH,
                )

    def test_corrupt_checkpoint_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "corrupt.pth"
            checkpoint_path.write_bytes(b"not a torch checkpoint")
            with self.assertRaisesRegex(
                ActionModelCompatibilityError,
                "Could not read LSTM checkpoint",
            ):
                load_action_model(
                    checkpoint_path,
                    "cpu",
                    SEQUENCE_LENGTH,
                )

    def test_incomplete_checkpoint_metadata_is_rejected(self):
        checkpoint = make_checkpoint()
        checkpoint.pop("sequence_length")
        with self.assertRaisesRegex(
            ActionModelCompatibilityError,
            "sequence_length",
        ):
            load_action_model_from_checkpoint(
                checkpoint,
                "cpu",
                SEQUENCE_LENGTH,
            )

    def test_sequence_length_and_feature_width_must_match(self):
        checkpoint = make_checkpoint()
        checkpoint["sequence_length"] = SEQUENCE_LENGTH - 1
        with self.assertRaisesRegex(ActionModelCompatibilityError, "sequence_length"):
            load_action_model_from_checkpoint(
                checkpoint,
                "cpu",
                SEQUENCE_LENGTH,
            )

        checkpoint = make_checkpoint()
        checkpoint["input_size"] = POSE_FEATURE_SIZE - 1
        with self.assertRaisesRegex(ActionModelCompatibilityError, "input_size"):
            load_action_model_from_checkpoint(
                checkpoint,
                "cpu",
                SEQUENCE_LENGTH,
            )

    def test_unidirectional_tensor_shapes_are_rejected(self):
        unidirectional = torch.nn.LSTM(
            POSE_FEATURE_SIZE,
            8,
            1,
            batch_first=True,
        )
        checkpoint = make_checkpoint()
        checkpoint["model_state_dict"] = unidirectional.state_dict()
        with self.assertRaisesRegex(ActionModelCompatibilityError, "tensor shapes"):
            load_action_model_from_checkpoint(
                checkpoint,
                "cpu",
                SEQUENCE_LENGTH,
            )

    def test_valid_checkpoint_loads_with_matching_output_map(self):
        model, inverse_map = load_action_model_from_checkpoint(
            make_checkpoint(),
            "cpu",
            SEQUENCE_LENGTH,
        )
        self.assertEqual(inverse_map, {0: "normal", 1: "fighting", 2: "shooting"})
        self.assertFalse(model.training)

    def test_saved_checkpoint_round_trips_through_runtime_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "compatible.pth"
            torch.save(make_checkpoint(), checkpoint_path)
            model, inverse_map = load_action_model(
                checkpoint_path,
                "cpu",
                SEQUENCE_LENGTH,
            )

        self.assertEqual(model.input_size, POSE_FEATURE_SIZE)
        self.assertEqual(inverse_map[2], "shooting")

    def test_classifier_rejects_incompatible_feature_shape(self):
        model, _ = load_action_model_from_checkpoint(
            make_checkpoint(),
            "cpu",
            SEQUENCE_LENGTH,
        )
        with self.assertRaisesRegex(ValueError, "Expected LSTM input shaped"):
            model(torch.zeros(1, SEQUENCE_LENGTH, POSE_FEATURE_SIZE - 1))


class ActionPredictionTests(unittest.TestCase):
    def test_prediction_is_not_called_until_sequence_is_complete(self):
        sequence = PoseSequenceBuffer(SEQUENCE_LENGTH)
        model = Mock()
        inverse_map = {0: "normal", 1: "fighting", 2: "shooting"}

        for _ in range(SEQUENCE_LENGTH - 1):
            sequence.append([0.0] * POSE_FEATURE_SIZE)
        self.assertIsNone(
            predict_action(model, sequence, inverse_map, "cpu", 0.85)
        )
        model.assert_not_called()

        sequence.append([0.0] * POSE_FEATURE_SIZE)
        model.return_value = torch.log(torch.tensor([[0.01, 0.98, 0.01]]))
        prediction = predict_action(
            model,
            sequence,
            inverse_map,
            "cpu",
            0.85,
        )
        self.assertEqual(prediction["class"], "fighting")
        self.assertEqual(prediction["class_id"], 1)
        self.assertAlmostEqual(prediction["confidence"], 0.98, places=5)
        self.assertEqual(tuple(model.call_args.args[0].shape), (1, 30, 132))

    def test_prediction_formatting_respects_threshold_and_shape(self):
        inverse_map = {0: "normal", 1: "fighting", 2: "shooting"}
        prediction = format_action_prediction(
            torch.log(torch.tensor([[0.01, 0.97, 0.02]])),
            inverse_map,
            0.85,
        )
        self.assertEqual(prediction["class"], "fighting")
        self.assertIsNone(
            format_action_prediction(
                torch.log(torch.tensor([[0.34, 0.33, 0.33]])),
                inverse_map,
                0.85,
            )
        )
        with self.assertRaisesRegex(ValueError, "one logits row"):
            format_action_prediction(
                torch.zeros(2, 3),
                inverse_map,
                0.85,
            )

    def test_yolo_frame_processing_continues_without_action_model(self):
        detection_result = type("Result", (), {"boxes": []})()
        detector = type(
            "Detector",
            (),
            {
                "class_names": {0: "person"},
                "predict": lambda self, frame, conf, imgsz, iou: [
                    detection_result
                ],
            },
        )()

        class EmptyTracker:
            def set_frame_rate(self, _frame_rate):
                return None

            def update(self, _detections):
                return []

        class NoAlertManager:
            def determine_threat(self, *_args):
                return "SAFE", (0, 255, 0)

            def log_threat(self, *_args, **_kwargs):
                return None

        state = {
            "person_durations": {},
            "unattended_bag_timer": 0.0,
            "frame_count": 0,
        }
        frame, messages = process_frame(
            np.zeros((48, 48, 3), dtype=np.uint8),
            detector,
            None,
            torch.device("cpu"),
            {},
            NoAlertManager(),
            EmptyTracker(),
            None,
            {},
            state_data=state,
        )

        self.assertEqual(frame.shape, (48, 48, 3))
        self.assertEqual(messages, [])
        self.assertEqual(state["frame_count"], 1)
        self.assertEqual(state["people_detected"], 0)
        self.assertEqual(state["action_detections"], [])

    def test_missing_lstm_skips_pose_but_keeps_person_tracking(self):
        box = type("Box", (), {})()
        box.xyxy = np.asarray([[4, 4, 40, 44]], dtype=np.float32)
        box.conf = np.asarray([0.91], dtype=np.float32)
        box.cls = np.asarray([0], dtype=np.float32)
        detection_result = type("Result", (), {"boxes": [box]})()
        detector = type(
            "Detector",
            (),
            {
                "class_names": {0: "person"},
                "predict": lambda self, frame, conf, imgsz, iou: [
                    detection_result
                ],
            },
        )()

        tracked_person = type(
            "TrackedPerson",
            (),
            {
                "xyxy": np.asarray([[4, 4, 40, 44]], dtype=np.float32),
                "tracker_id": np.asarray([17]),
                "confidence": np.asarray([0.91], dtype=np.float32),
                "__len__": lambda self: 1,
            },
        )()

        class SinglePersonTracker:
            def set_frame_rate(self, _frame_rate):
                return None

            def update(self, _detections):
                return tracked_person

        class NoAlertManager:
            def determine_threat(self, *_args):
                return "SAFE", (0, 255, 0)

            def log_threat(self, *_args, **_kwargs):
                return None

        pose_estimator = Mock()
        state = {
            "person_durations": {},
            "unattended_bag_timer": 0.0,
            "frame_count": 0,
        }
        frame, messages = process_frame(
            np.zeros((48, 48, 3), dtype=np.uint8),
            detector,
            None,
            torch.device("cpu"),
            {},
            NoAlertManager(),
            SinglePersonTracker(),
            pose_estimator,
            {},
            state_data=state,
        )

        self.assertEqual(frame.shape, (48, 48, 3))
        self.assertEqual(messages, [])
        self.assertEqual(state["people_detected"], 1)
        self.assertEqual(state["person_detections"][0]["track_id"], 17)
        self.assertEqual(state["person_detections"][0]["bbox"], [4, 4, 40, 44])
        pose_estimator.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()