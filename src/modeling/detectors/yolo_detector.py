import hashlib
import json
import logging
from pathlib import Path

import torch
from ultralytics import YOLO

from src.config import EXPECTED_DETECTOR_CLASSES, ROOT_DIR
from src.utils.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    def __init__(self):
        self.model = None
        self.class_names = None
        self.requested_model_path = None
        self.model_path = None
        self.is_custom_model = False
        self.is_validated = False
        self.alerts_enabled = False
        self.model_diagnostic = "Model has not been loaded."
        self.loaded_class_map = {}
        self.weapon_detection_available = False
        print("YOLODetector initialized.")

    def load_model(self, model_path):
        root_dir = Path(ROOT_DIR)
        requested_path = Path(model_path)

        if not requested_path.is_absolute():
            requested_path = root_dir / requested_path

        self.requested_model_path = requested_path
        fallback_path = root_dir / "yolo11s.pt"
        load_fallback = False
        fallback_reason = None

        if (
            not requested_path.is_file()
            or requested_path.stat().st_size < 10000
        ):
            fallback_reason = (
                "Configured detector checkpoint is missing or too small: "
                f"{requested_path}"
            )

            if not fallback_path.is_file():
                raise FileNotFoundError(
                    f"{fallback_reason}. No fallback checkpoint exists at "
                    f"{fallback_path}. Install valid trained weights or set "
                    "SMARTVISION_YOLO_MODEL_PATH."
                )

            logger.warning(
                "%s. Loading general-purpose fallback %s; it is not a "
                "weapon detector.",
                fallback_reason,
                fallback_path,
            )
            requested_path = fallback_path
            load_fallback = True

        logger.info("Loading YOLO checkpoint: %s", requested_path)

        custom_load_error = None
        try:
            model = YOLO(str(requested_path))
        except Exception as exc:
            if requested_path.resolve() == fallback_path.resolve():
                raise RuntimeError(
                    f"The fallback YOLO checkpoint could not be loaded: "
                    f"{fallback_path}: {exc}"
                ) from exc
            custom_load_error = str(exc)
            fallback_reason = (
                "Configured detector checkpoint could not be loaded: "
                f"{self.requested_model_path}: {custom_load_error}"
            )
            if not fallback_path.is_file():
                raise RuntimeError(
                    f"The configured YOLO checkpoint could not be loaded: "
                    f"{requested_path}: {exc}. No fallback checkpoint exists "
                    f"at {fallback_path}. Supply valid custom weights or "
                    "restore the root yolo11s.pt fallback."
                ) from exc
            logger.exception(
                "%s. Loading general-purpose fallback %s; it is not a "
                "weapon detector.",
                fallback_reason,
                fallback_path,
            )
            requested_path = fallback_path
            load_fallback = True
            try:
                model = YOLO(str(requested_path))
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Neither configured YOLO checkpoint "
                    f"{self.requested_model_path} nor fallback checkpoint "
                    f"{fallback_path} could be loaded. Custom error: "
                    f"{custom_load_error}; fallback error: {fallback_exc}"
                ) from fallback_exc
        self.model = model
        self.class_names = model.names
        self.model_path = requested_path
        model_names: object = model.names
        if isinstance(model_names, dict):
            self.loaded_class_map = {
                int(class_id): str(name).strip()
                for class_id, name in model_names.items()
            }
            loaded_names = tuple(
                name.lower()
                for _, name in sorted(self.loaded_class_map.items())
            )
        elif isinstance(model_names, (list, tuple)):
            self.loaded_class_map = {
                class_id: str(name).strip()
                for class_id, name in enumerate(model_names)
            }
            loaded_names = tuple(
                name.lower()
                for name in self.loaded_class_map.values()
            )
        else:
            self.loaded_class_map = {}
            loaded_names = ()
        self.is_custom_model = (
            requested_path.resolve() == self.requested_model_path.resolve()
            and loaded_names == EXPECTED_DETECTOR_CLASSES
        )
        self.weapon_detection_available = bool(
            self.is_custom_model
            and {"gun", "knife"}.issubset(set(loaded_names))
        )
        if self.is_custom_model:
            metadata_path = self.requested_model_path.with_suffix(
                self.requested_model_path.suffix + ".validation.json"
            )
            validation = {}
            try:
                validation = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                if not isinstance(validation, dict):
                    raise ValueError("Validation record must be a JSON object.")
                digest = hashlib.sha256()
                with self.requested_model_path.open("rb") as weights_file:
                    for chunk in iter(
                        lambda: weights_file.read(1024 * 1024),
                        b"",
                    ):
                        digest.update(chunk)
                self.is_validated = (
                    validation.get("classes") == list(EXPECTED_DETECTOR_CLASSES)
                    and validation.get("weights_sha256") == digest.hexdigest()
                    and validation.get("validation_completed") is True
                    and isinstance(validation.get("validation_metrics"), dict)
                    and isinstance(validation.get("validation_summary"), dict)
                    and all(
                        isinstance(
                            validation["validation_summary"].get(metric),
                            (int, float),
                        )
                        for metric in ("precision", "recall", "map50", "map50_95")
                    )
                    and validation.get("representative_validation_completed")
                    is True
                )
            except (OSError, ValueError, TypeError):
                self.is_validated = False
            self.alerts_enabled = (
                self.is_validated
                and validation.get("production_alerts_approved") is True
            )
            self.model_diagnostic = (
                "Validated six-class SmartVision detector loaded; production "
                "alerts are enabled."
                if self.alerts_enabled
                else (
                    "Six-class detector weights loaded. Production alerts "
                    "remain disabled until validation is complete and an "
                    "operator approves the validation record."
                    if self.is_validated
                    else (
                        "Six-class detector weights loaded, but a matching "
                        "validation record with metrics and representative "
                        "image tests was not found. Production alerts remain "
                        "disabled."
                    )
                )
            )
        else:
            missing_names = sorted(
                set(EXPECTED_DETECTOR_CLASSES) - set(loaded_names)
            )
            self.model_diagnostic = (
                "Active weights are not the configured six-class weapon "
                "detector. "
                "Missing project classes: "
                f"{', '.join(missing_names) if missing_names else 'class order mismatch'}."
            )
        if load_fallback:
            self.model_diagnostic = (
                f"{fallback_reason}. Active checkpoint: {requested_path}. "
                "The general-purpose COCO fallback has no gun, pistol, "
                "rifle, or project-trained weapon class; its generic knife "
                "class is not counted as a validated weapon. "
                + self.model_diagnostic
            )

        logger.info(
            "Loaded YOLO checkpoint %s with class map: %s",
            self.model_path,
            ", ".join(
                f"{class_id}={class_name}"
                for class_id, class_name in self.loaded_class_map.items()
            ),
        )
        if not self.weapon_detection_available:
            logger.error(
                "Weapon detection unavailable: %s",
                self.model_diagnostic,
            )

    def predict(self, frame, conf=0.5, imgsz=640, iou=0.7):
        if self.model is None:
            raise RuntimeError(
                "Model has not been loaded. Call load_model() first."
            )

        with torch.inference_mode():
            return self.model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                iou=iou,
                verbose=False,
            )