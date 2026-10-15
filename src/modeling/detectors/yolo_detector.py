import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from src.config import EXPECTED_DETECTOR_CLASSES, ROOT_DIR
from src.utils.base_detector import BaseDetector


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
        print("YOLODetector initialized.")

    def load_model(self, model_path):
        root_dir = Path(ROOT_DIR)
        requested_path = Path(model_path)

        if not requested_path.is_absolute():
            requested_path = root_dir / requested_path

        self.requested_model_path = requested_path
        fallback_path = root_dir / "yolo11s.pt"

        if (
            not requested_path.is_file()
            or requested_path.stat().st_size < 10000
        ):
            print(f"Custom model unavailable: {requested_path}")

            if not fallback_path.is_file():
                raise FileNotFoundError(
                    "No usable YOLO model was found.\n"
                    f"Custom model: {requested_path}\n"
                    f"Fallback model: {fallback_path}"
                )

            print(f"Using standard YOLO model: {fallback_path}")
            requested_path = fallback_path

        print(f"Loading YOLO model from: {requested_path}")

        model = YOLO(str(requested_path))
        self.model = model
        self.class_names = model.names
        self.model_path = requested_path
        model_names: object = model.names
        if isinstance(model_names, dict):
            loaded_names = tuple(
                str(name).strip().lower()
                for _, name in sorted(model_names.items())
            )
        elif isinstance(model_names, (list, tuple)):
            loaded_names = tuple(
                str(name).strip().lower() for name in model_names
            )
        else:
            loaded_names = ()
        self.is_custom_model = (
            requested_path.resolve() == self.requested_model_path.resolve()
            and loaded_names == EXPECTED_DETECTOR_CLASSES
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
                "Loaded weights are not the configured six-class detector. "
                "Missing project classes: "
                f"{', '.join(missing_names) if missing_names else 'class order mismatch'}."
            )

        print("YOLO model loaded successfully.")
        print(f"Classes found: {self.class_names}")

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