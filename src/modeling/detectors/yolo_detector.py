import logging
from pathlib import Path

import torch
from ultralytics import YOLO

from src.config import EXPECTED_DETECTOR_CLASSES, ROOT_DIR
from src.modeling.detectors.runtime_status import publish_detector_status
from src.modeling.detectors.validation import (
    sha256_file,
    validate_validation_record,
)
from src.utils.base_detector import BaseDetector


logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """
    SmartVision Pro YOLO detector.

    Safety behavior:
    - Uses the configured custom checkpoint when available.
    - Falls back to generic YOLO only for inference continuity.
    - Never treats generic YOLO weights as a validated weapon detector.
    - Enables production alerts only when the custom weights have a
      matching validation record and explicit operator approval.
    """

    MINIMUM_CHECKPOINT_SIZE = 10_000

    def __init__(self):
        self.model = None
        self.class_names = None

        self.requested_model_path = None
        self.model_path = None
        self.loaded_checkpoint_sha256 = None

        self.is_custom_model = False
        self.is_validated = False
        self.production_alerts_approved = False
        self.alerts_enabled = False
        self.weapon_detection_available = False

        self.loaded_class_map = {}
        self.model_diagnostic = "Model has not been loaded."
        self.validation_status = {
            "passed": False,
            "failures": ["Model has not been loaded."],
        }

        print("YOLODetector initialized.")

    @staticmethod
    def _checkpoint_is_available(path):
        """Return whether a checkpoint exists and is not suspiciously small."""
        try:
            return (
                path.is_file()
                and path.stat().st_size >= YOLODetector.MINIMUM_CHECKPOINT_SIZE
            )
        except OSError:
            return False

    @staticmethod
    def _get_class_map(model_names):
        """Normalize YOLO class names to a dictionary of integer IDs."""
        if isinstance(model_names, dict):
            return {
                int(class_id): str(name).strip()
                for class_id, name in model_names.items()
            }

        if isinstance(model_names, (list, tuple)):
            return {
                class_id: str(name).strip()
                for class_id, name in enumerate(model_names)
            }

        return {}

    def _load_validation_record(self):
        """
        Verify the validation record belonging to the requested checkpoint.

        A record must match the exact checkpoint and contain the expected
        classes, validation metrics, representative tests, and completion
        flags. A record alone does not prove real-world detector accuracy.
        """
        if self.requested_model_path is None:
            self.validation_status = {
                "passed": False,
                "failures": ["No custom checkpoint was requested."],
            }
            return {}

        self.validation_status = validate_validation_record(
            self.requested_model_path
        )
        self.is_validated = self.validation_status["passed"]
        if not self.is_validated:
            logger.warning(
                "Custom detector validation failed for %s: %s",
                self.requested_model_path,
                "; ".join(self.validation_status["failures"]),
            )
        return self.validation_status["record"]

    def load_model(self, model_path):
        """Load custom YOLO weights or use a clearly identified fallback."""
        root_dir = Path(ROOT_DIR).resolve()
        requested_path = Path(model_path).expanduser()

        if not requested_path.is_absolute():
            requested_path = root_dir / requested_path

        requested_path = requested_path.resolve()
        self.requested_model_path = requested_path

        fallback_path = (root_dir / "yolo11s.pt").resolve()

        self.model = None
        self.class_names = None
        self.model_path = None
        self.loaded_checkpoint_sha256 = None

        self.is_custom_model = False
        self.is_validated = False
        self.production_alerts_approved = False
        self.alerts_enabled = False
        self.weapon_detection_available = False

        self.loaded_class_map = {}
        self.model_diagnostic = ""
        self.validation_status = {
            "passed": False,
            "failures": ["Custom checkpoint is not loaded."],
        }
        publish_detector_status(self)

        load_fallback = False
        fallback_reason = None

        # -------------------------------------------------------------
        # CHECK CONFIGURED CHECKPOINT
        # -------------------------------------------------------------
        if not self._checkpoint_is_available(requested_path):
            fallback_reason = (
                "The configured custom detector checkpoint is missing "
                "or too small: "
                f"{requested_path}"
            )

            if not self._checkpoint_is_available(fallback_path):
                self.model_diagnostic = (
                    f"{fallback_reason}. The generic fallback is also "
                    f"unavailable: {fallback_path}. Restore a valid "
                    "checkpoint before running inference."
                )

                logger.error(self.model_diagnostic)
                publish_detector_status(self)

                raise FileNotFoundError(self.model_diagnostic)

            logger.warning(
                "%s. Loading generic YOLO for inference continuity. "
                "The fallback is not a validated SmartVision weapon detector.",
                fallback_reason,
            )

            requested_path = fallback_path
            load_fallback = True

        # -------------------------------------------------------------
        # LOAD MODEL
        # -------------------------------------------------------------
        try:
            logger.info("Loading YOLO checkpoint: %s", requested_path)
            model = YOLO(str(requested_path))

        except Exception as exc:
            if load_fallback:
                self.model_diagnostic = (
                    f"Unable to load the generic fallback checkpoint "
                    f"{fallback_path}: {exc}"
                )
                logger.exception(self.model_diagnostic)
                publish_detector_status(self)
                raise RuntimeError(self.model_diagnostic) from exc

            logger.exception(
                "Unable to load the configured custom checkpoint: %s",
                requested_path,
            )

            if not self._checkpoint_is_available(fallback_path):
                self.model_diagnostic = (
                    f"Custom checkpoint could not be loaded: {exc}. "
                    f"Generic fallback is unavailable: {fallback_path}"
                )
                publish_detector_status(self)
                raise RuntimeError(self.model_diagnostic) from exc

            fallback_reason = (
                f"Custom checkpoint could not be loaded: "
                f"{self.requested_model_path}. Error: {exc}"
            )

            logger.warning(
                "%s. Loading generic YOLO for inference continuity.",
                fallback_reason,
            )

            try:
                model = YOLO(str(fallback_path))
                requested_path = fallback_path
                load_fallback = True

            except Exception as fallback_exc:
                self.model_diagnostic = (
                    "Both the custom checkpoint and generic fallback failed "
                    "to load. "
                    f"Custom error: {exc}. Fallback error: {fallback_exc}"
                )
                logger.exception(self.model_diagnostic)
                publish_detector_status(self)
                raise RuntimeError(self.model_diagnostic) from fallback_exc

        self.model = model
        self.class_names = model.names
        self.model_path = requested_path
        if not load_fallback and requested_path == self.requested_model_path:
            try:
                self.loaded_checkpoint_sha256 = sha256_file(requested_path)
            except OSError:
                logger.exception(
                    "Could not hash the loaded custom detector checkpoint: %s",
                    requested_path,
                )

        # -------------------------------------------------------------
        # READ MODEL CLASS NAMES
        # -------------------------------------------------------------
        self.loaded_class_map = self._get_class_map(model.names)

        loaded_names = tuple(
            name.lower()
            for _, name in sorted(self.loaded_class_map.items())
        )

        # A generic fallback must never qualify as the requested custom
        # detector, even if a model happens to contain similarly named
        # classes.
        self.is_custom_model = bool(
            not load_fallback
            and requested_path == self.requested_model_path
            and loaded_names == EXPECTED_DETECTOR_CLASSES
        )

        self.weapon_detection_available = bool(
            self.is_custom_model
            and {"gun", "knife"}.issubset(set(loaded_names))
        )

        # -------------------------------------------------------------
        # VALIDATE CUSTOM DETECTOR
        # -------------------------------------------------------------
        if self.is_custom_model:
            self._load_validation_record()
            self.production_alerts_approved = bool(
                self.validation_status["production_alerts_approved"]
            )

            self.alerts_enabled = bool(
                self.is_validated
                and self.production_alerts_approved
            )

            if self.alerts_enabled:
                self.model_diagnostic = (
                    "Custom six-class detector loaded. A matching validation "
                    "record and explicit production-alert approval were found. "
                    "Check the live configuration before sending notifications."
                )

            elif self.is_validated:
                self.model_diagnostic = (
                    "Custom six-class detector loaded and its validation "
                    "record matches the checkpoint. Production alerts remain "
                    "disabled because explicit operator approval is absent."
                )

            else:
                failures = self.validation_status.get("failures", [])
                self.model_diagnostic = (
                    "Custom six-class detector loaded, but a valid matching "
                    "validation record was not found. Production alerts remain "
                    "disabled until validation and representative image tests "
                    "pass. "
                    + (
                        "Validation blockers: " + "; ".join(failures)
                        if failures
                        else ""
                    )
                )

        else:
            missing_names = sorted(
                set(EXPECTED_DETECTOR_CLASSES) - set(loaded_names)
            )

            if load_fallback:
                self.model_diagnostic = (
                    f"{fallback_reason or 'Custom checkpoint unavailable'}. "
                    f"Active checkpoint: {requested_path}. "
                    "This is a generic YOLO fallback, not a validated "
                    "SmartVision weapon detector. "
                    f"Missing project classes: "
                    f"{', '.join(missing_names) if missing_names else 'custom class mapping'}. "
                    "Production alerts remain disabled."
                )
            else:
                self.model_diagnostic = (
                    "The loaded checkpoint does not match the expected "
                    "SmartVision six-class detector. "
                    f"Missing project classes: "
                    f"{', '.join(missing_names) if missing_names else 'class order mismatch'}. "
                    "Production alerts remain disabled."
                )

        # -------------------------------------------------------------
        # FINAL SAFETY CHECK
        # -------------------------------------------------------------
        if not self.is_custom_model:
            self.is_validated = False
            self.alerts_enabled = False

        if not self.weapon_detection_available:
            self.alerts_enabled = False

        publish_detector_status(self)

        logger.info(
            "Loaded YOLO checkpoint: %s",
            self.model_path,
        )

        logger.info(
            "YOLO class map: %s",
            ", ".join(
                f"{class_id}={class_name}"
                for class_id, class_name in self.loaded_class_map.items()
            ),
        )

        logger.info(
            "Custom model: %s | Validated: %s | Weapon classes available: %s "
            "| Production alerts approved: %s",
            self.is_custom_model,
            self.is_validated,
            self.weapon_detection_available,
            self.alerts_enabled,
        )

        if not self.weapon_detection_available:
            logger.error(
                "Custom weapon detection is unavailable: %s",
                self.model_diagnostic,
            )

        return self.model

    def predict(self, frame, conf=0.5, imgsz=640, iou=0.7):
        """Run YOLO inference on an image or video frame."""
        if self.model is None:
            raise RuntimeError(
                "YOLO model has not been loaded. Call load_model() first."
            )

        if frame is None:
            raise ValueError("Cannot run YOLO inference on an empty frame.")

        with torch.inference_mode():
            return self.model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                iou=iou,
                verbose=False,
            )