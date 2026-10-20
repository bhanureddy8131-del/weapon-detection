"""Evaluate the configured custom detector and write an auditable report."""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPECTED_DETECTOR_CLASSES, ROOT_DIR, YOLO_MODEL_PATH
from src.modeling.detectors.validation import (
    DEFAULT_VALIDATION_DATA,
    configured_thresholds,
    configured_representative_limits,
    identify_validation_dataset,
    sha256_file,
)


WEAPON_CLASSES = {"gun", "knife"}


def _iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _class_names(names):
    if isinstance(names, dict):
        try:
            return [
                str(names[index]).strip().lower()
                for index in range(len(names))
            ]
        except KeyError:
            return []
    if isinstance(names, (list, tuple)):
        return [str(name).strip().lower() for name in names]
    return []


def _numeric_metric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _result_metric(metrics, name):
    return _numeric_metric(getattr(metrics.box, name, None))


def _array_metric(values, index):
    try:
        return _numeric_metric(values[index])
    except (IndexError, TypeError):
        return None


def extract_validation_metrics(metrics, class_names):
    """Extract aggregate and class-level box metrics from Ultralytics results."""
    box_metrics = metrics.box
    class_indices = list(getattr(box_metrics, "ap_class_index", []))
    class_positions = {
        int(class_id): position
        for position, class_id in enumerate(class_indices)
    }
    per_class = {}
    for class_id, class_name in enumerate(class_names):
        position = class_positions.get(class_id)
        per_class[class_name] = {
            "precision": (
                _array_metric(getattr(box_metrics, "p", None), position)
                if position is not None else None
            ),
            "recall": (
                _array_metric(getattr(box_metrics, "r", None), position)
                if position is not None else None
            ),
            "map50": (
                _array_metric(getattr(box_metrics, "ap50", None), position)
                if position is not None else None
            ),
            "map50_95": (
                _array_metric(getattr(box_metrics, "ap", None), position)
                if position is not None else None
            ),
        }

    summary = {
        "precision": _result_metric(metrics, "mp"),
        "recall": _result_metric(metrics, "mr"),
        "map50": _result_metric(metrics, "map50"),
        "map50_95": _result_metric(metrics, "map"),
    }
    raw_results = {}
    for key, value in getattr(metrics, "results_dict", {}).items():
        number = _numeric_metric(value)
        if number is not None:
            raw_results[str(key)] = number
    confusion_matrix = None
    matrix = getattr(
        getattr(metrics, "confusion_matrix", None),
        "matrix",
        None,
    )
    if matrix is not None:
        try:
            confusion_matrix = np.asarray(matrix).tolist()
        except (TypeError, ValueError):
            confusion_matrix = None
    return summary, per_class, raw_results, confusion_matrix


def measure_inference_latency(
    model,
    image_paths,
    image_size,
    confidence,
    device,
    runs,
):
    """Measure per-image predict latency separately from dataset evaluation."""
    if not image_paths:
        return {
            "completed": False,
            "mean_ms": None,
            "p95_ms": None,
            "samples": 0,
        }
    samples = []
    for image_path in image_paths:
        for _ in range(2):
            model.predict(
                source=str(image_path),
                imgsz=image_size,
                conf=confidence,
                device=device,
                verbose=False,
            )
        for _ in range(runs):
            started = time.perf_counter()
            model.predict(
                source=str(image_path),
                imgsz=image_size,
                conf=confidence,
                device=device,
                verbose=False,
            )
            samples.append((time.perf_counter() - started) * 1000)
    return {
        "completed": bool(samples),
        "runs_per_image": runs,
        "image_count": len(image_paths),
        "sample_count": len(samples),
        "device": str(device),
        "image_size": image_size,
        "confidence_threshold": confidence,
        "mean_ms": float(np.mean(samples)) if samples else None,
        "p50_ms": float(np.percentile(samples, 50)) if samples else None,
        "p95_ms": float(np.percentile(samples, 95)) if samples else None,
    }


def _representative_paths(dataset, data_yaml, samples_per_class):
    yaml_path = Path(data_yaml).resolve()
    with yaml_path.open("r", encoding="utf-8-sig") as file:
        configuration = yaml.safe_load(file)
    root = Path(configuration.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()

    selected = {}
    candidates = {}
    for image_path in dataset["image_paths"]:
        image_parts = list(image_path.parts)
        image_component = next(
            (
                index for index in range(len(image_parts) - 1, -1, -1)
                if image_parts[index].lower() == "images"
            ),
            None,
        )
        label_path = (
            Path(*image_parts[:image_component], "labels",
                 *image_parts[image_component + 1:]).with_suffix(".txt")
            if image_component is not None
            else image_path.with_suffix(".txt")
        )
        if not label_path.is_file():
            continue
        found = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 5:
                try:
                    class_id = int(fields[0])
                except ValueError:
                    continue
                if 0 <= class_id < len(EXPECTED_DETECTOR_CLASSES):
                    found.add(class_id)
        for class_id in found:
            class_paths = candidates.setdefault(class_id, [])
            if len(class_paths) < samples_per_class:
                class_paths.append(image_path)

    for class_id, class_name in enumerate(EXPECTED_DETECTOR_CLASSES):
        selected[class_name] = candidates.get(class_id, [])
    return selected, root


def _label_path(image_path):
    parts = list(image_path.parts)
    image_component = next(
        (
            index for index in range(len(parts) - 1, -1, -1)
            if parts[index].lower() == "images"
        ),
        None,
    )
    if image_component is None:
        return image_path.with_suffix(".txt")
    return Path(
        *parts[:image_component],
        "labels",
        *parts[image_component + 1:],
    ).with_suffix(".txt")


def _ground_truth(image_path, image):
    height, width = image.shape[:2]
    label_path = _label_path(image_path)
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        try:
            class_id = int(values[0])
            center_x, center_y, box_width, box_height = map(float, values[1:5])
        except ValueError:
            continue
        boxes.append((
            class_id,
            np.asarray([
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ], dtype=np.float64),
        ))
    return boxes


def _iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_representative_images(
    model,
    dataset,
    data_yaml,
    image_size,
    confidence,
    device,
    samples_per_class=1,
):
    selected, dataset_root = _representative_paths(
        dataset, data_yaml, samples_per_class
    )
    selected_classes = {
        class_name for class_name, paths in selected.items() if paths
    }
    unique_images = sorted({
        path for paths in selected.values() for path in paths
    })
    results = []
    missing = set(EXPECTED_DETECTOR_CLASSES) - selected_classes
    if not unique_images:
        return {
            "completed": False,
            "tested_classes": sorted(selected_classes),
            "results": [],
            "missed_weapon_detections": 0,
            "weapon_false_positives": 0,
            "missing_classes": sorted(missing),
        }

    predictions = model.predict(
        source=[str(path) for path in unique_images],
        imgsz=image_size,
        conf=confidence,
        device=device,
        verbose=False,
    )
    predictions_by_path = {
        Path(result.path).resolve(): result
        for result in predictions
    }
    missed_weapons = 0
    weapon_false_positives = 0
    image_classes = {}
    for class_name, paths in selected.items():
        for image_path in paths:
            image_classes.setdefault(image_path, set()).add(class_name)

    for image_path in unique_images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not decode representative image: {image_path}")
        result = predictions_by_path.get(image_path.resolve())
        if result is None:
            raise ValueError(f"No prediction result was returned for {image_path}")

        truths = _ground_truth(image_path, image)
        predicted_boxes = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                predicted_boxes.append({
                    "class_id": int(box.cls[0].item()),
                    "confidence": float(box.conf[0].item()),
                    "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                })

        matched_prediction_indices = set()
        per_class_results = {}
        for class_id, class_name in enumerate(EXPECTED_DETECTOR_CLASSES):
            class_truths = [
                box for target_id, box in truths if target_id == class_id
            ]
            candidate_predictions = [
                (index, prediction)
                for index, prediction in enumerate(predicted_boxes)
                if prediction["class_id"] == class_id
            ]
            candidate_predictions.sort(
                key=lambda item: item[1]["confidence"],
                reverse=True,
            )
            matched_truths = set()
            matches = 0
            for prediction_index, prediction in candidate_predictions:
                best_index = None
                best_iou = 0.5
                for truth_index, truth in enumerate(class_truths):
                    if truth_index in matched_truths:
                        continue
                    overlap = _iou(prediction["xyxy"], truth)
                    if overlap >= best_iou:
                        best_iou = overlap
                        best_index = truth_index
                if best_index is not None:
                    matched_truths.add(best_index)
                    matched_prediction_indices.add(prediction_index)
                    matches += 1

            missed = len(class_truths) - matches
            per_class_results[class_name] = {
                "ground_truth": len(class_truths),
                "predictions": len(candidate_predictions),
                "matched": matches,
                "missed": missed,
                "false_positives": len(candidate_predictions) - matches,
            }
            if class_name in WEAPON_CLASSES:
                missed_weapons += missed
                weapon_false_positives += (
                    len(candidate_predictions) - matches
                )

        results.append({
            "image": str(image_path.resolve()),
            "selected_for_classes": sorted(image_classes[image_path]),
            "per_class": per_class_results,
            "weapon_detections": [
                item for item in predicted_boxes
                if EXPECTED_DETECTOR_CLASSES[item["class_id"]]
                in WEAPON_CLASSES
            ],
            "unmatched_weapon_predictions": [
                item for index, item in enumerate(predicted_boxes)
                if (
                    EXPECTED_DETECTOR_CLASSES[item["class_id"]]
                    in WEAPON_CLASSES
                    and index not in matched_prediction_indices
                )
            ],
        })

    return {
        "completed": not missing,
        "tested_classes": sorted(selected_classes),
        "results": results,
        "missed_weapon_detections": missed_weapons,
        "weapon_false_positives": weapon_false_positives,
        "missing_classes": sorted(missing),
        "iou_threshold": 0.5,
        "confidence_threshold": confidence,
        "dataset_root": str(dataset_root),
    }


def _metric_checks(summary, per_class, thresholds):
    checks = {}
    for metric in ("precision", "recall", "map50", "map50_95"):
        value = summary.get(metric)
        threshold = thresholds[metric]
        checks[metric] = {
            "passed": value is not None and value >= threshold,
            "actual": value,
            "minimum": threshold,
        }
    for class_name in ("gun", "knife", "person"):
        metric_name = f"{class_name}_recall"
        value = per_class.get(class_name, {}).get("recall")
        threshold = thresholds[metric_name]
        checks[metric_name] = {
            "passed": value is not None and value >= threshold,
            "actual": value,
            "minimum": threshold,
        }
    return checks


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _report_directory(split):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(ROOT_DIR) / "runs" / "validation" / split / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _configure_logging(report_directory):
    log_path = Path(ROOT_DIR) / "logs" / "detector_validation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("Validation run output: %s", report_directory)
    return log_path


def _prepare_thresholds(arguments):
    thresholds = configured_thresholds()
    representative_limits = configured_representative_limits()
    for name in (
        "precision",
        "recall",
        "map50",
        "map50_95",
        "gun_recall",
        "knife_recall",
        "person_recall",
    ):
        value = getattr(arguments, f"min_{name}")
        if value < thresholds[name]:
            raise ValueError(
                f"--min-{name.replace('_', '-')} cannot be below the "
                "configured SMARTVISION_VALIDATION minimum."
            )
        thresholds[name] = value
    for argument_name, threshold_name in (
        ("max_weapon_missed_detections", "max_weapon_missed_detections"),
        ("max_weapon_false_positives", "max_weapon_false_positives"),
    ):
        value = getattr(arguments, argument_name)
        if value < 0:
            raise ValueError(f"--{argument_name.replace('_', '-')} cannot be negative.")
        if value > representative_limits[threshold_name]:
            raise ValueError(
                f"--{argument_name.replace('_', '-')} cannot relax the "
                "configured SMARTVISION_VALIDATION representative limit."
            )
        thresholds[threshold_name] = value
    return thresholds


def run_validation(arguments):
    checkpoint = Path(arguments.checkpoint).expanduser()
    data_yaml = Path(arguments.data).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(ROOT_DIR) / checkpoint
    if not data_yaml.is_absolute():
        data_yaml = Path(ROOT_DIR) / data_yaml
    checkpoint = checkpoint.resolve()
    data_yaml = data_yaml.resolve()
    report_directory = _report_directory(arguments.split)
    log_path = _configure_logging(report_directory)
    timestamp = _iso_now()
    report = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": None,
        "expected_classes": list(EXPECTED_DETECTOR_CLASSES),
        "actual_classes": None,
        "validation_dataset": None,
        "validation_started_at": timestamp,
        "validation_completed_at": None,
        "evaluation_split": arguments.split,
        "evaluation_settings": {
            "image_size": arguments.image_size,
            "batch": arguments.batch,
            "device": arguments.device,
            "validation_confidence": 0.001,
            "validation_iou": 0.6,
            "representative_confidence": arguments.representative_confidence,
            "representative_iou": 0.5,
            "representatives_per_class": arguments.representatives_per_class,
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
        },
        "validation_completed": False,
        "validation_passed": False,
        "validation_metrics": {},
        "confusion_matrix": None,
        "validation_summary": {},
        "per_class_metrics": {},
        "acceptance_thresholds": {},
        "acceptance_checks": {},
        "representative_tests": {
            "completed": False,
            "tested_classes": [],
            "results": [],
            "missed_weapon_detections": 0,
            "weapon_false_positives": 0,
        },
        "inference_latency": {
            "completed": False,
            "mean_ms": None,
            "p95_ms": None,
            "samples": 0,
        },
        "checks": {},
        "production_alerts_approved": False,
        "log_path": str(log_path),
        "report_directory": str(report_directory),
    }
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".validation.json")
    checkpoint_loaded = False
    evaluation_completed = False

    try:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Configured checkpoint is missing: {checkpoint}")
        if checkpoint.stat().st_size < 10_000:
            raise ValueError(f"Checkpoint is too small to be valid: {checkpoint}")
        report["checkpoint_sha256"] = sha256_file(checkpoint)

        thresholds = _prepare_thresholds(arguments)
        report["acceptance_thresholds"] = thresholds
        dataset = identify_validation_dataset(data_yaml, arguments.split)
        report["validation_dataset"] = {
            key: value for key, value in dataset.items()
            if key != "image_paths"
        }
        if dataset["class_names"] != list(EXPECTED_DETECTOR_CLASSES):
            raise ValueError(
                "Validation dataset class names do not match the six-class "
                f"contract: {dataset['class_names']}"
            )
        if dataset["configured_nc"] != len(EXPECTED_DETECTOR_CLASSES):
            raise ValueError(
                "Validation dataset nc does not match the six-class contract."
            )

        logging.info("Loading actual configured checkpoint: %s", checkpoint)
        model = YOLO(str(checkpoint))
        checkpoint_loaded = True
        report["actual_classes"] = _class_names(model.names)
        if report["actual_classes"] != list(EXPECTED_DETECTOR_CLASSES):
            raise ValueError(
                "Checkpoint class names do not match the expected six-class "
                f"mapping: {report['actual_classes']}"
            )

        logging.info(
            "Evaluating split=%s, images=%d, data=%s",
            arguments.split,
            dataset["image_count"],
            data_yaml,
        )
        validation_metrics = model.val(
            data=str(data_yaml),
            split=arguments.split,
            imgsz=arguments.image_size,
            batch=arguments.batch,
            conf=0.001,
            iou=0.6,
            device=arguments.device,
            project=str(report_directory),
            name="ultralytics",
            exist_ok=True,
            plots=True,
            verbose=True,
        )
        evaluation_completed = True
        summary, per_class, raw_metrics, confusion_matrix = extract_validation_metrics(
            validation_metrics,
            report["actual_classes"],
        )
        report["validation_summary"] = summary
        report["per_class_metrics"] = per_class
        report["validation_metrics"] = raw_metrics
        report["confusion_matrix"] = confusion_matrix
        report["confusion_matrix_image"] = str(
            Path(getattr(
                validation_metrics,
                "save_dir",
                report_directory / "ultralytics",
            )) / "confusion_matrix.png"
        )
        report["ultralytics_results_directory"] = str(
            getattr(validation_metrics, "save_dir", report_directory / "ultralytics")
        )

        logging.info("Running representative labeled-image checks.")
        report["representative_tests"] = evaluate_representative_images(
            model=model,
            dataset=dataset,
            data_yaml=data_yaml,
            image_size=arguments.image_size,
            confidence=arguments.representative_confidence,
            device=arguments.device,
            samples_per_class=arguments.representatives_per_class,
        )
        latency_paths = [
            Path(item["image"])
            for item in report["representative_tests"]["results"]
        ]
        report["inference_latency"] = measure_inference_latency(
            model=model,
            image_paths=latency_paths,
            image_size=arguments.image_size,
            confidence=arguments.representative_confidence,
            device=arguments.device,
            runs=arguments.latency_runs,
        )
        report["validation_completed_at"] = _iso_now()
        report["validation_completed"] = True

        acceptance_checks = _metric_checks(summary, per_class, thresholds)
        rep_tests = report["representative_tests"]
        acceptance_checks["representative_tests_complete"] = {
            "passed": (
                rep_tests["completed"]
                and set(EXPECTED_DETECTOR_CLASSES).issubset(
                    set(rep_tests["tested_classes"])
                )
            ),
            "actual": rep_tests["tested_classes"],
            "expected": list(EXPECTED_DETECTOR_CLASSES),
        }
        acceptance_checks["no_missed_weapon_detections"] = {
            "passed": (
                rep_tests["missed_weapon_detections"]
                <= thresholds["max_weapon_missed_detections"]
            ),
            "actual": rep_tests["missed_weapon_detections"],
            "maximum": thresholds["max_weapon_missed_detections"],
        }
        acceptance_checks["weapon_false_positive_limit"] = {
            "passed": (
                rep_tests["weapon_false_positives"]
                <= thresholds["max_weapon_false_positives"]
            ),
            "actual": rep_tests["weapon_false_positives"],
            "maximum": thresholds["max_weapon_false_positives"],
        }
        report["acceptance_checks"] = acceptance_checks
        metrics_present = bool(raw_metrics) and all(
            value is not None for value in summary.values()
        ) and all(
            value is not None
            for class_metrics in per_class.values()
            for value in class_metrics.values()
        )
        representative_passed = all(
            check["passed"]
            for name, check in acceptance_checks.items()
            if name.startswith("representative")
            or name.startswith("no_missed")
            or name.startswith("weapon_false")
        )
        check_values = {
            "checkpoint_exists": checkpoint.is_file(),
            "checkpoint_identity": (
                checkpoint.is_file()
                and sha256_file(checkpoint) == report["checkpoint_sha256"]
            ),
            "checkpoint_loadable": checkpoint_loaded,
            "class_mapping": (
                report["actual_classes"] == list(EXPECTED_DETECTOR_CLASSES)
            ),
            "dataset_available": bool(dataset["image_count"]),
            "dataset_identity": (
                report["validation_dataset"].get("dataset_id")
                == dataset.get("dataset_id")
                and report["validation_dataset"].get("dataset_yaml_sha256")
                == dataset.get("dataset_yaml_sha256")
                and report["validation_dataset"].get("configured_nc")
                == len(EXPECTED_DETECTOR_CLASSES)
                and report["validation_dataset"].get("class_names")
                == list(EXPECTED_DETECTOR_CLASSES)
            ),
            "evaluation_settings": (
                arguments.split == "val"
                and arguments.image_size >= 1
                and arguments.batch >= 1
                and arguments.representatives_per_class >= 1
                and isinstance(arguments.device, str)
                and bool(arguments.device.strip())
            ),
            "evaluation_completed": evaluation_completed,
            "metrics_present": metrics_present,
            "acceptance_thresholds": all(
                check["passed"]
                for name, check in acceptance_checks.items()
                if name not in {
                    "representative_tests_complete",
                    "no_missed_weapon_detections",
                    "weapon_false_positive_limit",
                }
            ),
            "representative_tests": representative_passed,
        }
        report["checks"] = {
            name: {"passed": passed}
            for name, passed in check_values.items()
        }
        report["validation_passed"] = all(check_values.values())
        if sha256_file(checkpoint) != report["checkpoint_sha256"]:
            raise RuntimeError(
                "Checkpoint contents changed during evaluation; refusing to "
                "bind this report to a different file."
            )
        if arguments.split == "val":
            sidecar_record = dict(report)
            sidecar_record.pop("log_path", None)
            sidecar_record.pop("report_directory", None)
            _write_json(sidecar_path, sidecar_record)
            logging.info("Validation sidecar written: %s", sidecar_path)
        else:
            logging.info(
                "Final held-out test report will not replace the validation sidecar."
            )

    except Exception as exc:
        logging.exception("Detector validation failed: %s", exc)
        report["failure"] = f"{type(exc).__name__}: {exc}"
        if report["validation_completed_at"] is None:
            report["validation_completed_at"] = _iso_now()
        report["checks"] = {
            "checkpoint_exists": {
                "passed": checkpoint.is_file(),
            },
            "checkpoint_identity": {"passed": False},
            "checkpoint_loadable": {"passed": checkpoint_loaded},
            "class_mapping": {
                "passed": report["actual_classes"]
                == list(EXPECTED_DETECTOR_CLASSES),
            },
            "dataset_available": {
                "passed": report["validation_dataset"] is not None,
            },
            "dataset_identity": {"passed": False},
            "evaluation_settings": {"passed": False},
            "evaluation_completed": {"passed": evaluation_completed},
            "metrics_present": {"passed": False},
            "acceptance_thresholds": {"passed": False},
            "representative_tests": {"passed": False},
        }

    report_name = (
        "validation_report.json"
        if arguments.split == "val"
        else "final_test_report.json"
    )
    timestamped_report = report_directory / report_name
    latest_report = Path(ROOT_DIR) / "runs" / "validation" / "latest"
    try:
        _write_json(timestamped_report, report)
        latest_report_name = (
            "validation_report.json"
            if arguments.split == "val"
            else "final_test_report.json"
        )
        _write_json(latest_report / latest_report_name, report)
    except (OSError, TypeError, ValueError):
        logging.exception("Could not persist validation report.")
        return 2

    logging.info(
        "Validation result: %s",
        "PASS" if report["validation_passed"] else "FAIL",
    )
    print(f"Configured checkpoint: {checkpoint}")
    print(f"Validation report: {latest_report / latest_report_name}")
    print(f"Validation log: {log_path}")
    print(f"Validation passed: {report['validation_passed']}")
    if report.get("validation_summary"):
        print(
            "Metrics:",
            json.dumps(report["validation_summary"], sort_keys=True),
        )
    return 0 if report["validation_passed"] else 1


def build_parser():
    defaults = configured_thresholds()
    representative_limits = configured_representative_limits()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(YOLO_MODEL_PATH))
    parser.add_argument("--data", type=Path, default=DEFAULT_VALIDATION_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--representatives-per-class", type=int, default=1)
    parser.add_argument("--representative-confidence", type=float, default=0.25)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--min-precision", type=float, default=defaults["precision"])
    parser.add_argument("--min-recall", type=float, default=defaults["recall"])
    parser.add_argument("--min-map50", type=float, default=defaults["map50"])
    parser.add_argument("--min-map50-95", type=float, default=defaults["map50_95"])
    parser.add_argument("--min-gun-recall", type=float, default=defaults["gun_recall"])
    parser.add_argument("--min-knife-recall", type=float, default=defaults["knife_recall"])
    parser.add_argument("--min-person-recall", type=float, default=defaults["person_recall"])
    parser.add_argument(
        "--max-weapon-missed-detections",
        type=int,
        default=representative_limits["max_weapon_missed_detections"],
    )
    parser.add_argument(
        "--max-weapon-false-positives",
        type=int,
        default=representative_limits["max_weapon_false_positives"],
    )
    return parser


def main():
    arguments = build_parser().parse_args()
    if arguments.image_size <= 0 or arguments.batch <= 0:
        raise SystemExit("--image-size and --batch must be positive.")
    if arguments.representatives_per_class < 1:
        raise SystemExit("--representatives-per-class must be at least 1.")
    if arguments.latency_runs < 1:
        raise SystemExit("--latency-runs must be at least 1.")
    return run_validation(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
