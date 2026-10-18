"""Report local SmartVision runtime and run timed YOLO inference on one image."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import mediapipe
import numpy as np
import supervision
import torch
import ultralytics
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EXPECTED_DETECTOR_CLASSES,
    LSTM_MODEL_PATH,
    POSE_MODEL_PATH,
    ROOT_DIR,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL_PATH,
)
from src.modeling.detectors.yolo_detector import YOLODetector


def _device_name():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_sample_image():
    image_dir = Path(ROOT_DIR) / "The_final-4" / "test" / "images"
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not image_dir.is_dir():
        return None
    return next(
        (path for path in sorted(image_dir.iterdir())
         if path.is_file() and path.suffix.lower() in image_suffixes),
        None,
    )


def _class_names(detector):
    names = detector.class_names or {}
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def _read_dataset_classes():
    dataset_yaml = Path(ROOT_DIR) / "The_final-4" / "data.yaml"
    if not dataset_yaml.is_file():
        return None
    with dataset_yaml.open("r", encoding="utf-8-sig") as yaml_file:
        configuration = yaml.safe_load(yaml_file)
    names = configuration.get("names", {})
    if isinstance(names, dict):
        return [str(names[index]).strip().lower() for index in sorted(names)]
    if isinstance(names, list):
        return [str(name).strip().lower() for name in names]
    return None


def run_diagnostics(image_path=None, runs=10, image_size=YOLO_IMAGE_SIZE, confidence=0.1):
    if runs < 1:
        raise ValueError("runs must be at least 1")

    custom_path = Path(YOLO_MODEL_PATH)
    lstm_path = Path(LSTM_MODEL_PATH)
    pose_path = Path(POSE_MODEL_PATH)
    fallback_path = Path(ROOT_DIR) / "yolo11s.pt"
    detector = YOLODetector()
    detector.load_model(custom_path)

    selected_image = Path(image_path) if image_path else _default_sample_image()
    image_result = {
        "path": str(selected_image) if selected_image else None,
        "readable": False,
        "shape": None,
        "detection_count": None,
        "mean_inference_ms": None,
        "p95_inference_ms": None,
        "detections": [],
    }
    if selected_image is not None:
        image = cv2.imread(str(selected_image))
        if image is None:
            raise ValueError(f"Could not decode image: {selected_image}")
        image_result["readable"] = True
        image_result["shape"] = list(image.shape)

        for _ in range(min(2, runs)):
            detector.predict(image, conf=confidence, imgsz=image_size)
        timings = []
        results = None
        for _ in range(runs):
            started = time.perf_counter()
            results = detector.predict(
                image,
                conf=confidence,
                imgsz=image_size,
            )
            timings.append((time.perf_counter() - started) * 1000)

        detections = []
        boxes = results[0].boxes if results else None
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0].item())
                coordinates = [float(value) for value in box.xyxy[0].tolist()]
                detections.append({
                    "class_id": class_id,
                    "class": detector.class_names.get(class_id, "Unknown")
                    if isinstance(detector.class_names, dict)
                    else detector.class_names[class_id],
                    "confidence": float(box.conf[0].item()),
                    "xyxy": coordinates,
                })
        image_result.update({
            "detection_count": len(detections),
            "mean_inference_ms": statistics.mean(timings),
            "p95_inference_ms": float(np.percentile(timings, 95)),
            "detections": detections,
        })

    dataset_classes = _read_dataset_classes()
    report = {
        "python": sys.version.split()[0],
        "versions": {
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "mediapipe": mediapipe.__version__,
            "supervision": supervision.__version__,
        },
        "compute_device": _device_name(),
        "checkpoints": {
            "configured_yolo": str(custom_path),
            "configured_yolo_exists": custom_path.is_file(),
            "loaded_yolo": str(detector.model_path),
            "loaded_yolo_is_custom": detector.is_custom_model,
            "loaded_yolo_is_validated": detector.is_validated,
            "generic_fallback": str(fallback_path),
            "generic_fallback_exists": fallback_path.is_file(),
            "lstm": str(lstm_path),
            "lstm_exists": lstm_path.is_file(),
            "mediapipe_pose": str(pose_path),
            "mediapipe_pose_exists": pose_path.is_file(),
        },
        "detector": {
            "classes": _class_names(detector),
            "expected_project_classes": list(EXPECTED_DETECTOR_CLASSES),
            "dataset_classes": dataset_classes,
            "dataset_matches_project_contract": (
                dataset_classes == list(EXPECTED_DETECTOR_CLASSES)
                if dataset_classes is not None
                else None
            ),
            "alerts_enabled": detector.alerts_enabled,
            "diagnostic": detector.model_diagnostic,
        },
        "inference": {
            "image_size": image_size,
            "confidence_threshold": confidence,
            "runs": runs,
            "sample": image_result,
        },
        "limitations": [
            "Timing is a runtime measurement, not an accuracy evaluation.",
            "A generic COCO model does not support the project's gun or ATM classes.",
            "LSTM recognition is unavailable when its configured checkpoint is missing.",
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Image to run YOLO inference on")
    parser.add_argument("--runs", type=int, default=10, help="Timed inference runs")
    parser.add_argument("--image-size", type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument("--confidence", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(
        run_diagnostics(args.image, args.runs, args.image_size, args.confidence),
        indent=2,
    ))


if __name__ == "__main__":
    main()
