import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from scripts import approve_detector_alerts
from src.config import EXPECTED_DETECTOR_CLASSES
from src.modeling.detectors.validation import (
    configured_thresholds,
    identify_validation_dataset,
    validation_report_sha256,
    validate_validation_record,
)
from src.modeling.detectors.yolo_detector import YOLODetector


class DetectorValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.checkpoint = self.root / "custom.pt"
        self.checkpoint.write_bytes(b"custom checkpoint bytes" * 1000)
        self.dataset_yaml = self.root / "dataset.yaml"
        image_directory = self.root / "valid" / "images"
        label_directory = self.root / "valid" / "labels"
        image_directory.mkdir(parents=True)
        label_directory.mkdir(parents=True)
        self.representative_images = {}
        for class_id, class_name in enumerate(EXPECTED_DETECTOR_CLASSES):
            image_path = image_directory / f"representative-{class_name}.jpg"
            image_path.write_bytes(f"labeled {class_name} image".encode())
            (label_directory / image_path.with_suffix(".txt").name).write_text(
                f"{class_id} 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            self.representative_images[class_name] = image_path
        self.dataset_yaml.write_text(
            yaml.safe_dump({
                "path": str(self.root),
                "val": "valid/images",
                "nc": 6,
                "names": {
                    index: name
                    for index, name in enumerate(EXPECTED_DETECTOR_CLASSES)
                },
            }),
            encoding="utf-8",
        )
        self.sidecar = self.checkpoint.with_suffix(".pt.validation.json")
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        self.record = self.valid_record()
        self.write_record()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def valid_record(self):
        thresholds = configured_thresholds()
        thresholds.update({
            "max_weapon_missed_detections": 0,
            "max_weapon_false_positives": 0,
        })
        dataset = identify_validation_dataset(self.dataset_yaml)
        dataset.pop("image_paths")
        summary = {
            "precision": 0.9,
            "recall": 0.9,
            "map50": 0.9,
            "map50_95": 0.8,
        }
        per_class = {
            name: {
                "precision": 0.9,
                "recall": 0.9,
                "map50": 0.9,
                "map50_95": 0.8,
            }
            for name in EXPECTED_DETECTOR_CLASSES
        }
        checks = {
            name: {"passed": True}
            for name in (
                "checkpoint_exists",
                "checkpoint_identity",
                "checkpoint_loadable",
                "class_mapping",
                "dataset_available",
                "dataset_identity",
                "evaluation_settings",
                "evaluation_completed",
                "metrics_present",
                "acceptance_thresholds",
                "representative_tests",
            )
        }
        acceptance_checks = {
            name: {
                "passed": True,
                "actual": (
                    summary[name]
                    if name in summary
                    else per_class[name.removesuffix("_recall")]["recall"]
                ),
                "minimum": thresholds[name],
            }
            for name in configured_thresholds()
        }
        acceptance_checks.update({
            "representative_tests_complete": {
                "passed": True,
                "actual": list(EXPECTED_DETECTOR_CLASSES),
                "expected": list(EXPECTED_DETECTOR_CLASSES),
            },
            "no_missed_weapon_detections": {
                "passed": True,
                "actual": 0,
                "maximum": 0,
            },
            "weapon_false_positive_limit": {
                "passed": True,
                "actual": 0,
                "maximum": 0,
            },
        })
        representative_results = []
        for class_name, image_path in self.representative_images.items():
            per_image = {}
            for class_id, result_class in enumerate(EXPECTED_DETECTOR_CLASSES):
                count = int(class_name == result_class)
                per_image[result_class] = {
                    "ground_truth": count,
                    "predictions": count,
                    "matched": count,
                    "missed": 0,
                    "false_positives": 0,
                }
            representative_results.append({
                "image": str(image_path.resolve()),
                "selected_for_classes": [class_name],
                "per_class": per_image,
            })
        dataset_path = self.root / "valid" / "images"
        return {
            "schema_version": 1,
            "checkpoint_path": str(self.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(
                self.checkpoint.read_bytes()
            ).hexdigest(),
            "expected_classes": list(EXPECTED_DETECTOR_CLASSES),
            "actual_classes": list(EXPECTED_DETECTOR_CLASSES),
            "validation_dataset": dataset,
            "validation_started_at": self.now.isoformat(),
            "validation_completed_at": self.now.isoformat(),
            "evaluation_split": "val",
            "evaluation_settings": {
                "image_size": 640,
                "batch": 8,
                "device": "cpu",
                "validation_confidence": 0.001,
                "validation_iou": 0.6,
                "representative_confidence": 0.25,
                "representative_iou": 0.5,
                "representatives_per_class": 1,
            },
            "validation_completed": True,
            "validation_passed": True,
            "validation_metrics": {
                "metrics/precision(B)": 0.9,
                "metrics/recall(B)": 0.9,
            },
            "validation_summary": summary,
            "per_class_metrics": per_class,
            "acceptance_thresholds": thresholds,
            "acceptance_checks": acceptance_checks,
            "representative_tests": {
                "completed": True,
                "tested_classes": list(EXPECTED_DETECTOR_CLASSES),
                "results": representative_results,
                "missed_weapon_detections": 0,
                "weapon_false_positives": 0,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.5,
                "dataset_root": str(self.root),
            },
            "checks": checks,
            "production_alerts_approved": False,
        }

    def write_record(self):
        self.sidecar.write_text(
            json.dumps(self.record),
            encoding="utf-8",
        )

    def validate(self):
        return validate_validation_record(
            self.checkpoint,
            data_yaml=self.dataset_yaml,
            now=self.now,
        )

    def approve_record(self):
        self.record.update({
            "production_alerts_approved": True,
            "production_alerts_approved_by": "operator",
            "production_alerts_approved_at": (
                self.now + timedelta(seconds=1)
            ).isoformat(),
            "production_alerts_approved_checkpoint_sha256": (
                self.record["checkpoint_sha256"]
            ),
            "production_alerts_test_provider_accepted": True,
            "production_alerts_test_provider_accepted_at": (
                self.now + timedelta(seconds=1)
            ).isoformat(),
        })
        self.record["production_alerts_validation_sha256"] = (
            validation_report_sha256(self.record)
        )

    def test_missing_checkpoint_is_rejected(self):
        self.checkpoint.unlink()
        self.assertFalse(self.validate()["passed"])

    def test_generic_yolo_class_map_is_not_a_custom_weapon_detector(self):
        generic_model = SimpleNamespace(names={0: "person", 43: "knife"})
        with (
            patch("src.modeling.detectors.yolo_detector.ROOT_DIR", str(self.root)),
            patch(
                "src.modeling.detectors.runtime_status.STATUS_PATH",
                self.root / "logs" / "detector_runtime_status.json",
            ),
            patch(
                "src.modeling.detectors.yolo_detector.YOLO",
                return_value=generic_model,
            ),
        ):
            detector = YOLODetector()
            detector.load_model(self.checkpoint)

        self.assertFalse(detector.is_custom_model)
        self.assertFalse(detector.is_validated)
        self.assertFalse(detector.alerts_enabled)

    def test_missing_validation_record_is_rejected(self):
        self.sidecar.unlink()
        self.assertFalse(self.validate()["passed"])

    def test_incorrect_checkpoint_sha256_is_rejected(self):
        self.record["checkpoint_sha256"] = "0" * 64
        self.write_record()
        status = self.validate()
        self.assertFalse(status["passed"])
        self.assertTrue(any("SHA-256" in item for item in status["failures"]))

    def test_incorrect_class_names_are_rejected(self):
        self.record["actual_classes"][2] = "pistol"
        self.write_record()
        self.assertFalse(self.validate()["passed"])

    def test_missing_metrics_are_rejected(self):
        del self.record["validation_summary"]["recall"]
        self.write_record()
        self.assertFalse(self.validate()["passed"])

    def test_missing_evaluation_settings_are_rejected(self):
        del self.record["evaluation_settings"]
        self.write_record()
        self.assertFalse(self.validate()["passed"])

    def test_changed_dataset_manifest_is_rejected(self):
        self.record["validation_dataset"]["manifest_sha256"] = "0" * 64
        self.write_record()
        status = self.validate()
        self.assertFalse(status["passed"])
        self.assertTrue(any("manifest_sha256" in item for item in status["failures"]))

    def test_failed_acceptance_threshold_is_rejected(self):
        self.record["validation_summary"]["map50"] = 0.1
        self.record["acceptance_checks"]["map50"].update({
            "passed": False,
            "actual": 0.1,
        })
        self.write_record()
        self.assertFalse(self.validate()["passed"])

    def test_inconsistent_representative_totals_are_rejected(self):
        self.record["representative_tests"]["missed_weapon_detections"] = 1
        self.write_record()
        self.assertFalse(self.validate()["passed"])

    def test_incomplete_representative_tests_are_rejected(self):
        self.record["representative_tests"]["tested_classes"].remove("knife")
        self.write_record()
        self.assertFalse(self.validate()["passed"])

    def test_stale_validation_record_is_rejected(self):
        self.record["validation_completed_at"] = (
            self.now - timedelta(days=31)
        ).isoformat()
        self.write_record()
        self.assertFalse(self.validate()["passed"])

    def test_validated_checkpoint_without_operator_approval_remains_unapproved(self):
        status = self.validate()
        self.assertTrue(status["passed"], status["failures"])
        self.assertFalse(status["production_alerts_approved"])

    def test_validated_checkpoint_requires_explicit_approval_metadata(self):
        self.approve_record()
        self.write_record()
        status = self.validate()
        self.assertTrue(status["passed"], status["failures"])
        self.assertTrue(status["production_alerts_approved"])

    def test_approval_without_operator_metadata_is_not_accepted(self):
        self.record["production_alerts_approved"] = True
        self.write_record()
        status = self.validate()
        self.assertTrue(status["passed"], status["failures"])
        self.assertFalse(status["production_alerts_approved"])

    def test_approval_requires_an_authorized_account(self):
        with (
            patch(
                "scripts.approve_detector_alerts.getpass.getuser",
                return_value="intruder",
            ),
            patch.object(
                approve_detector_alerts.settings,
                "ALERT_APPROVERS",
                frozenset({"operator"}),
            ),
            patch.object(
                approve_detector_alerts,
                "ALERT_ENDPOINT_VERIFIED",
                True,
            ),
            patch.object(
                approve_detector_alerts.ntfy_service,
                "notification_state",
                return_value=("ready", None),
            ),
            patch.object(
                approve_detector_alerts.ntfy_service,
                "send_ntfy_notification",
            ) as send_notification,
        ):
            with self.assertRaises(PermissionError):
                approve_detector_alerts.approve_production_alerts(
                    self.checkpoint,
                    approve_detector_alerts.CONFIRMATION,
                )
        send_notification.assert_not_called()

    def test_provider_failure_does_not_record_approval(self):
        status = {
            "passed": True,
            "production_alerts_approved": False,
            "record": self.record,
            "validation_record_path": str(self.sidecar),
            "failures": [],
        }
        with (
            patch(
                "scripts.approve_detector_alerts.getpass.getuser",
                return_value="operator",
            ),
            patch.object(
                approve_detector_alerts.settings,
                "ALERT_APPROVERS",
                frozenset({"operator"}),
            ),
            patch.object(
                approve_detector_alerts,
                "ALERT_ENDPOINT_VERIFIED",
                True,
            ),
            patch.object(
                approve_detector_alerts.ntfy_service,
                "notification_state",
                return_value=("ready", None),
            ),
            patch.object(
                approve_detector_alerts,
                "validate_validation_record",
                return_value=status,
            ),
            patch.object(
                approve_detector_alerts.ntfy_service,
                "send_ntfy_notification",
                side_effect=approve_detector_alerts.ntfy_service.NtfyDeliveryError(
                    "test provider failure"
                ),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "approval was not recorded"):
                approve_detector_alerts.approve_production_alerts(
                    self.checkpoint,
                    approve_detector_alerts.CONFIRMATION,
                )
        persisted = json.loads(self.sidecar.read_text(encoding="utf-8"))
        self.assertFalse(persisted["production_alerts_approved"])

    def test_successful_approval_records_hashes_after_mocked_provider_acceptance(self):
        status = {
            "passed": True,
            "production_alerts_approved": False,
            "record": self.record,
            "validation_record_path": str(self.sidecar),
            "failures": [],
        }
        with (
            patch(
                "scripts.approve_detector_alerts.getpass.getuser",
                return_value="operator",
            ),
            patch.object(
                approve_detector_alerts.settings,
                "ALERT_APPROVERS",
                frozenset({"operator"}),
            ),
            patch.object(
                approve_detector_alerts,
                "ALERT_ENDPOINT_VERIFIED",
                True,
            ),
            patch.object(
                approve_detector_alerts.ntfy_service,
                "notification_state",
                return_value=("ready", None),
            ),
            patch.object(
                approve_detector_alerts,
                "validate_validation_record",
                return_value=status,
            ),
            patch.object(
                approve_detector_alerts.ntfy_service,
                "send_ntfy_notification",
            ) as send_notification,
        ):
            approve_detector_alerts.approve_production_alerts(
                self.checkpoint,
                approve_detector_alerts.CONFIRMATION,
            )

        send_notification.assert_called_once()
        approved_status = self.validate()
        self.assertTrue(approved_status["passed"], approved_status["failures"])
        self.assertTrue(approved_status["production_alerts_approved"])

    def test_detector_keeps_validation_and_operator_approval_separate(self):
        fake_model = SimpleNamespace(
            names={
                index: name
                for index, name in enumerate(EXPECTED_DETECTOR_CLASSES)
            }
        )
        with (
            patch(
                "src.modeling.detectors.validation.DEFAULT_VALIDATION_DATA",
                self.dataset_yaml,
            ),
            patch(
                "src.modeling.detectors.runtime_status.STATUS_PATH",
                self.root / "logs" / "detector_runtime_status.json",
            ),
            patch(
                "src.modeling.detectors.yolo_detector.ROOT_DIR",
                str(self.root),
            ),
            patch(
                "src.modeling.detectors.yolo_detector.YOLO",
                return_value=fake_model,
            ),
        ):
            detector = YOLODetector()
            detector.load_model(self.checkpoint)
            self.assertTrue(detector.is_custom_model)
            self.assertTrue(detector.is_validated)
            self.assertFalse(detector.production_alerts_approved)
            self.assertFalse(detector.alerts_enabled)

            self.approve_record()
            self.write_record()
            approved_detector = YOLODetector()
            approved_detector.load_model(self.checkpoint)

        self.assertTrue(approved_detector.is_validated)
        self.assertTrue(approved_detector.production_alerts_approved)
        self.assertTrue(approved_detector.alerts_enabled)


if __name__ == "__main__":
    unittest.main()
