"""Shared detector-validation record and dataset checks."""

import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from src.config import (
    EXPECTED_DETECTOR_CLASSES,
    VALIDATION_DATA_PATH,
)


DEFAULT_VALIDATION_DATA = Path(VALIDATION_DATA_PATH)
VALIDATION_MAX_AGE = timedelta(days=30)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
REQUIRED_CHECKS = {
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
}
DEFAULT_THRESHOLDS = {
    "precision": 0.70,
    "recall": 0.70,
    "map50": 0.70,
    "map50_95": 0.50,
    "gun_recall": 0.70,
    "knife_recall": 0.70,
    "person_recall": 0.70,
}
METRIC_THRESHOLD_ENV = {
    "precision": "SMARTVISION_VALIDATION_MIN_PRECISION",
    "recall": "SMARTVISION_VALIDATION_MIN_RECALL",
    "map50": "SMARTVISION_VALIDATION_MIN_MAP50",
    "map50_95": "SMARTVISION_VALIDATION_MIN_MAP50_95",
    "gun_recall": "SMARTVISION_VALIDATION_MIN_GUN_RECALL",
    "knife_recall": "SMARTVISION_VALIDATION_MIN_KNIFE_RECALL",
    "person_recall": "SMARTVISION_VALIDATION_MIN_PERSON_RECALL",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_report_sha256(record):
    """Hash immutable evaluation content, excluding operator-approval fields."""
    evaluation = dict(record)
    for field in (
        "production_alerts_approved",
        "production_alerts_approved_by",
        "production_alerts_approved_at",
        "production_alerts_approved_checkpoint_sha256",
        "production_alerts_validation_sha256",
        "production_alerts_test_provider_accepted",
        "production_alerts_test_provider_accepted_at",
    ):
        evaluation.pop(field, None)
    encoded = json.dumps(
        evaluation,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configured_thresholds():
    thresholds = dict(DEFAULT_THRESHOLDS)
    for name, environment_variable in METRIC_THRESHOLD_ENV.items():
        raw_value = os.getenv(environment_variable)
        if raw_value is not None:
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{environment_variable} must be a finite value in [0, 1]."
                )
            thresholds[name] = value
    return thresholds


def configured_representative_limits():
    limits = {
        "max_weapon_missed_detections": 0,
        "max_weapon_false_positives": 0,
    }
    for name in limits:
        environment_variable = f"SMARTVISION_VALIDATION_{name.upper()}"
        raw_value = os.getenv(environment_variable)
        if raw_value is not None:
            try:
                value = int(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"{environment_variable} must be a non-negative integer."
                ) from exc
            if value < 0:
                raise ValueError(
                    f"{environment_variable} must be a non-negative integer."
                )
            limits[name] = value
    return limits


def _dataset_names(configuration):
    names = configuration.get("names")
    if isinstance(names, dict):
        try:
            return [
                str(names[index]).strip().lower()
                for index in range(len(names))
            ]
        except KeyError:
            return []
    if isinstance(names, list):
        return [str(name).strip().lower() for name in names]
    return []


def _resolve_dataset_path(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def identify_validation_dataset(data_yaml, split="val"):
    """Identify the configured held-out split using its YAML and file manifest."""
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset configuration is missing: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8-sig") as file:
        configuration = yaml.safe_load(file)
    if not isinstance(configuration, dict):
        raise ValueError("Dataset configuration must be a YAML mapping.")

    names = _dataset_names(configuration)
    configured_nc = configuration.get("nc")
    if configured_nc != len(EXPECTED_DETECTOR_CLASSES):
        raise ValueError(
            "Dataset nc does not match the six-class detector contract."
        )
    if names != list(EXPECTED_DETECTOR_CLASSES):
        raise ValueError(
            "Dataset class names do not match the six-class detector contract."
        )
    root = Path(configuration.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()

    split_value = configuration.get(split)
    if isinstance(split_value, list):
        image_dirs = [_resolve_dataset_path(root, item) for item in split_value]
    elif isinstance(split_value, str):
        image_dirs = [_resolve_dataset_path(root, split_value)]
    else:
        raise ValueError(f"Dataset configuration has no {split!r} split.")

    image_paths = []
    for image_dir in image_dirs:
        if image_dir.is_dir():
            image_paths.extend(
                path for path in image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise ValueError(f"The {split!r} dataset split contains no images.")

    yaml_digest = sha256_file(yaml_path)
    manifest = hashlib.sha256()
    label_count = 0
    for image_path in image_paths:
        try:
            relative_path = image_path.relative_to(root)
        except ValueError:
            relative_path = image_path

        label_parts = list(image_path.parts)
        image_component = next(
            (
                index for index in range(len(label_parts) - 1, -1, -1)
                if label_parts[index].lower() == "images"
            ),
            None,
        )
        if image_component is not None:
            label_parts[image_component] = "labels"
            label_path = Path(*label_parts).with_suffix(".txt")
        else:
            label_path = image_path.with_suffix(".txt")

        image_stat = image_path.stat()
        manifest.update(
            f"{relative_path}|{image_stat.st_size}|{image_stat.st_mtime_ns}\n"
            .encode("utf-8")
        )
        if label_path.is_file():
            label_count += 1
            label_stat = label_path.stat()
            try:
                relative_label = label_path.relative_to(root)
            except ValueError:
                relative_label = label_path
            manifest.update(
                f"label:{relative_label}|{label_stat.st_size}|"
                f"{label_stat.st_mtime_ns}\n".encode("utf-8")
            )
        else:
            manifest.update(f"label:{label_path.name}|missing\n".encode("utf-8"))

    dataset_id_hash = hashlib.sha256()
    dataset_id_hash.update(str(yaml_path).encode("utf-8"))
    dataset_id_hash.update(yaml_digest.encode("ascii"))
    dataset_id_hash.update(split.encode("utf-8"))
    dataset_id_hash.update(manifest.hexdigest().encode("ascii"))

    return {
        "dataset_yaml": str(yaml_path),
        "dataset_yaml_sha256": yaml_digest,
        "dataset_root": str(root),
        "configured_nc": configured_nc,
        "split": split,
        "image_count": len(image_paths),
        "label_file_count": label_count,
        "manifest_sha256": manifest.hexdigest(),
        "dataset_id": dataset_id_hash.hexdigest(),
        "class_names": names,
        "image_paths": image_paths,
    }


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _parse_timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return None
    return timestamp.astimezone(timezone.utc)


def validate_validation_record(
    checkpoint_path,
    record_path=None,
    data_yaml=None,
    now=None,
):
    """Return strict validation and production-approval state for a sidecar."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    data_yaml = DEFAULT_VALIDATION_DATA if data_yaml is None else data_yaml
    sidecar = (
        Path(record_path).expanduser().resolve()
        if record_path is not None
        else checkpoint.with_suffix(checkpoint.suffix + ".validation.json")
    )
    failures = []
    record = {}
    if not checkpoint.is_file():
        failures.append("Configured checkpoint is missing.")
    if not sidecar.is_file():
        failures.append("Checkpoint validation sidecar is missing.")
    else:
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Validation sidecar could not be read: {exc}")

    if not isinstance(record, dict):
        record = {}
        failures.append("Validation sidecar must contain a JSON object.")
    if record.get("schema_version") != 1:
        failures.append("Validation sidecar schema version is missing or unsupported.")

    if record.get("checkpoint_path") != str(checkpoint):
        failures.append("Validation record is for another checkpoint path.")

    if checkpoint.is_file():
        try:
            checkpoint_hash_matches = (
                record.get("checkpoint_sha256") == sha256_file(checkpoint)
            )
            if not checkpoint_hash_matches:
                failures.append("Checkpoint SHA-256 does not match the record.")
        except OSError as exc:
            checkpoint_hash_matches = False
            failures.append(f"Checkpoint SHA-256 could not be read: {exc}")
    else:
        checkpoint_hash_matches = False

    expected_classes = list(EXPECTED_DETECTOR_CLASSES)
    if record.get("expected_classes") != expected_classes:
        failures.append("Validation record has an incorrect expected class map.")
    if record.get("actual_classes") != expected_classes:
        failures.append("Checkpoint class names do not match the expected map.")

    completed_at = _parse_timestamp(record.get("validation_completed_at"))
    started_at = _parse_timestamp(record.get("validation_started_at"))
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    current_time = current_time.astimezone(timezone.utc)
    if started_at is None:
        failures.append("Validation start timestamp is missing or invalid.")
    if completed_at is None:
        failures.append("Validation completion timestamp is missing or invalid.")
    elif completed_at > current_time + timedelta(minutes=5):
        failures.append("Validation completion timestamp is in the future.")
    elif current_time - completed_at > VALIDATION_MAX_AGE:
        failures.append("Validation record is stale.")
    if started_at is not None and completed_at is not None:
        if started_at > completed_at:
            failures.append("Validation start timestamp is after completion.")
        if started_at > current_time + timedelta(minutes=5):
            failures.append("Validation start timestamp is in the future.")

    try:
        dataset = identify_validation_dataset(data_yaml, "val")
        record_dataset = record.get("validation_dataset")
        dataset_identity_matches = True
        if not isinstance(record_dataset, dict):
            failures.append("Validation dataset identification is missing.")
            dataset_identity_matches = False
        else:
            for key in (
                "dataset_yaml",
                "dataset_yaml_sha256",
                "dataset_root",
                "configured_nc",
                "split",
                "image_count",
                "label_file_count",
                "manifest_sha256",
                "dataset_id",
                "class_names",
            ):
                if record_dataset.get(key) != dataset.get(key):
                    dataset_identity_matches = False
                    failures.append(
                        f"Validation dataset does not match current {key}."
                    )
            if dataset["class_names"] != expected_classes:
                dataset_identity_matches = False
                failures.append("Validation dataset class names are incorrect.")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        dataset = None
        dataset_identity_matches = False
        failures.append(f"Validation dataset is unavailable: {exc}")

    evaluation_settings_valid = True
    if record.get("evaluation_split") != "val":
        evaluation_settings_valid = False
        failures.append("Validation report does not identify the val split.")
    evaluation_settings = record.get("evaluation_settings")
    if not isinstance(evaluation_settings, dict):
        evaluation_settings = {}
        evaluation_settings_valid = False
        failures.append("Validation evaluation settings are missing.")
    for name in ("image_size", "batch", "representatives_per_class"):
        value = evaluation_settings.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            evaluation_settings_valid = False
            failures.append(
                f"Validation evaluation setting {name} is missing or invalid."
            )
    if not isinstance(evaluation_settings.get("device"), str) or not (
        evaluation_settings.get("device", "").strip()
    ):
        evaluation_settings_valid = False
        failures.append("Validation evaluation setting device is missing or invalid.")
    for name in (
        "validation_confidence",
        "validation_iou",
        "representative_confidence",
        "representative_iou",
    ):
        value = evaluation_settings.get(name)
        if not _number(value):
            evaluation_settings_valid = False
            failures.append(
                f"Validation evaluation setting {name} is missing or invalid."
            )

    summary = record.get("validation_summary")
    if not isinstance(summary, dict):
        summary = {}
        failures.append("Validation metrics summary is missing.")
    required_metrics = ("precision", "recall", "map50", "map50_95")
    if any(not _number(summary.get(metric)) for metric in required_metrics):
        failures.append("One or more required aggregate metrics are missing or invalid.")
    raw_metrics = record.get("validation_metrics")
    if (
        not isinstance(raw_metrics, dict)
        or not raw_metrics
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_metrics.values()
        )
    ):
        failures.append("Raw validation metrics are missing.")

    per_class = record.get("per_class_metrics")
    if not isinstance(per_class, dict):
        per_class = {}
    for class_name in expected_classes:
        class_metrics = per_class.get(class_name)
        if not isinstance(class_metrics, dict) or any(
            not _number(class_metrics.get(metric))
            for metric in ("precision", "recall", "map50", "map50_95")
        ):
            failures.append(
                f"Per-class validation metrics are missing for {class_name}."
            )

    thresholds = record.get("acceptance_thresholds")
    if not isinstance(thresholds, dict):
        thresholds = {}
        failures.append("Acceptance thresholds are missing.")
    try:
        configured = configured_thresholds()
    except ValueError as exc:
        failures.append(f"Validation threshold configuration is invalid: {exc}")
        configured = dict(DEFAULT_THRESHOLDS)
    try:
        representative_limits = configured_representative_limits()
    except ValueError as exc:
        failures.append(f"Representative threshold configuration is invalid: {exc}")
        representative_limits = {
            "max_weapon_missed_detections": 0,
            "max_weapon_false_positives": 0,
        }
    threshold_names = tuple(configured)
    for metric in threshold_names:
        threshold = thresholds.get(metric)
        if not _number(threshold):
            failures.append(f"Acceptance threshold for {metric} is missing or invalid.")
        elif threshold < configured[metric]:
            failures.append(
                f"Recorded {metric} threshold is below the configured minimum."
            )
        elif metric in summary and summary[metric] < threshold:
            failures.append(f"Validation threshold failed for {metric}.")

    for class_name in ("gun", "knife", "person"):
        threshold_name = f"{class_name}_recall"
        class_metrics = per_class.get(class_name, {})
        if not isinstance(class_metrics, dict):
            class_metrics = {}
        if (
            _number(thresholds.get(threshold_name))
            and _number(class_metrics.get("recall"))
            and class_metrics["recall"] < thresholds[threshold_name]
        ):
            failures.append(f"Validation threshold failed for {threshold_name}.")

    acceptance_checks = record.get("acceptance_checks")
    if not isinstance(acceptance_checks, dict):
        acceptance_checks = {}
        failures.append("Explicit acceptance check results are missing.")
    required_acceptance_checks = {
        *threshold_names,
        "representative_tests_complete",
        "no_missed_weapon_detections",
        "weapon_false_positive_limit",
    }
    metric_check_values = []
    for check_name in required_acceptance_checks:
        check = acceptance_checks.get(check_name)
        if not isinstance(check, dict) or check.get("passed") is not True:
            failures.append(
                f"Acceptance check {check_name} did not explicitly pass."
            )
        if check_name in threshold_names and isinstance(check, dict):
            if check_name in summary:
                actual = summary.get(check_name)
            else:
                class_name = check_name.removesuffix("_recall")
                actual = per_class.get(class_name, {}).get("recall")
            threshold = thresholds.get(check_name)
            expected_passed = (
                _number(actual)
                and _number(threshold)
                and actual >= threshold
            )
            if (
                check.get("actual") != actual
                or check.get("minimum") != threshold
                or check.get("passed") is not expected_passed
            ):
                failures.append(
                    f"Acceptance check {check_name} does not match recorded metrics."
                )
            metric_check_values.append(expected_passed is True)

    representative = record.get("representative_tests")
    if not isinstance(representative, dict):
        representative = {}
        failures.append("Representative validation results are missing.")
    else:
        tested_classes = representative.get("tested_classes")
        results = representative.get("results")
        if (
            representative.get("completed") is not True
            or not isinstance(tested_classes, list)
            or not all(isinstance(name, str) for name in tested_classes)
            or not set(expected_classes).issubset(set(
                name for name in tested_classes if isinstance(name, str)
            ))
            or not isinstance(results, list)
            or not results
        ):
            failures.append("Representative validation tests are incomplete.")
        image_paths = (
            {path.resolve() for path in dataset.get("image_paths", [])}
            if isinstance(dataset, dict)
            else set()
        )
        reported_images = set()
        reported_test_classes = set()
        reported_missed_weapons = 0
        reported_weapon_false_positives = 0
        valid_result_count = 0
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    failures.append(
                        "Representative validation contains an invalid image result."
                    )
                    continue
                try:
                    result_image = Path(result.get("image", "")).expanduser().resolve()
                except (OSError, TypeError, ValueError):
                    result_image = None
                if result_image not in image_paths:
                    failures.append(
                        "Representative result image is not in the validation dataset."
                    )
                if result_image in reported_images:
                    failures.append(
                        "Representative validation contains a duplicate image result."
                    )
                reported_images.add(result_image)
                selected_classes = result.get("selected_for_classes")
                per_image = result.get("per_class")
                if (
                    not isinstance(selected_classes, list)
                    or not selected_classes
                    or any(name not in expected_classes for name in selected_classes)
                    or not isinstance(per_image, dict)
                ):
                    failures.append(
                        "Representative image class results are missing or invalid."
                    )
                    continue
                reported_test_classes.update(selected_classes)
                valid_image_result = True
                for class_name in expected_classes:
                    class_result = per_image.get(class_name)
                    if not isinstance(class_result, dict):
                        valid_image_result = False
                        break
                    counts = [
                        class_result.get(name)
                        for name in (
                            "ground_truth",
                            "predictions",
                            "matched",
                            "missed",
                            "false_positives",
                        )
                    ]
                    if any(
                        not isinstance(count, int)
                        or isinstance(count, bool)
                        or count < 0
                        for count in counts
                    ):
                        valid_image_result = False
                        break
                    ground_truth, predictions, matched, missed, false_positives = counts
                    if (
                        ground_truth != matched + missed
                        or predictions != matched + false_positives
                        or (class_name in selected_classes and ground_truth == 0)
                    ):
                        valid_image_result = False
                        break
                    if class_name in ("gun", "knife"):
                        reported_missed_weapons += missed
                        reported_weapon_false_positives += false_positives
                if not valid_image_result:
                    failures.append(
                        "Representative per-class detection counts are inconsistent."
                    )
                else:
                    valid_result_count += 1
        if valid_result_count != len(results or []):
            failures.append(
                "One or more representative image results are incomplete."
            )
        if reported_test_classes != set(tested_classes or []):
            failures.append(
                "Representative tested classes do not match the image results."
            )
        for count_name in (
            "missed_weapon_detections",
            "weapon_false_positives",
        ):
            count = representative.get(count_name)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                failures.append(
                    f"Representative result {count_name} is missing or invalid."
                )
        if (
            representative.get("missed_weapon_detections")
            != reported_missed_weapons
        ):
            failures.append(
                "Representative missed-weapon total does not match image results."
            )
        if (
            representative.get("weapon_false_positives")
            != reported_weapon_false_positives
        ):
            failures.append(
                "Representative false-positive total does not match image results."
            )
        max_missed = thresholds.get("max_weapon_missed_detections")
        max_false_positives = thresholds.get("max_weapon_false_positives")
        valid_missed_limit = (
            isinstance(max_missed, int)
            and not isinstance(max_missed, bool)
            and 0 <= max_missed
            <= representative_limits["max_weapon_missed_detections"]
        )
        valid_false_positive_limit = (
            isinstance(max_false_positives, int)
            and not isinstance(max_false_positives, bool)
            and 0 <= max_false_positives
            <= representative_limits["max_weapon_false_positives"]
        )
        if not valid_missed_limit:
            failures.append("Representative missed-detection limit is invalid.")
            max_missed = 0
        if not valid_false_positive_limit:
            failures.append("Representative false-positive limit is invalid.")
            max_false_positives = 0
        if (
            isinstance(representative.get("missed_weapon_detections"), int)
            and representative["missed_weapon_detections"] > max_missed
        ):
            failures.append("Representative tests missed a weapon detection.")
        if (
            isinstance(representative.get("weapon_false_positives"), int)
            and representative["weapon_false_positives"]
            > max_false_positives
        ):
            failures.append("Representative tests exceeded the false-positive limit.")

        if (
            evaluation_settings.get("representative_confidence")
            != representative.get("confidence_threshold")
            or evaluation_settings.get("representative_iou")
            != representative.get("iou_threshold")
        ):
            failures.append(
                "Representative tests do not match the recorded evaluation settings."
            )

        representative_checks = {
            "representative_tests_complete": (
                representative.get("completed") is True
                and set(expected_classes).issubset(reported_test_classes)
            ),
            "no_missed_weapon_detections": (
                isinstance(representative.get("missed_weapon_detections"), int)
                and not isinstance(
                    representative.get("missed_weapon_detections"), bool
                )
                and representative["missed_weapon_detections"]
                <= max_missed
            ),
            "weapon_false_positive_limit": (
                isinstance(representative.get("weapon_false_positives"), int)
                and not isinstance(
                    representative.get("weapon_false_positives"), bool
                )
                and representative["weapon_false_positives"]
                <= max_false_positives
            ),
        }
        representative_check_values = {
            "representative_tests_complete": {
                "actual": tested_classes,
                "expected": expected_classes,
            },
            "no_missed_weapon_detections": {
                "actual": representative.get("missed_weapon_detections"),
                "maximum": max_missed,
            },
            "weapon_false_positive_limit": {
                "actual": representative.get("weapon_false_positives"),
                "maximum": max_false_positives,
            },
        }
        for check_name, expected_passed in representative_checks.items():
            check = acceptance_checks.get(check_name)
            if (
                not isinstance(check, dict)
                or check.get("passed") is not expected_passed
                or any(
                    check.get(key) != value
                    for key, value in representative_check_values[
                        check_name
                    ].items()
                )
            ):
                failures.append(
                    f"Acceptance check {check_name} does not match representative results."
                )

    checks = record.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    for check_name in REQUIRED_CHECKS:
        check = checks.get(check_name)
        if not isinstance(check, dict) or check.get("passed") is not True:
            failures.append(f"Required validation check {check_name} did not pass.")

    if isinstance(checks, dict):
        required_check_values = {
            "checkpoint_exists": checkpoint.is_file(),
            "checkpoint_identity": (
                checkpoint.is_file()
                and record.get("checkpoint_path") == str(checkpoint)
                and checkpoint_hash_matches
            ),
            "checkpoint_loadable": checks.get("checkpoint_loadable", {}).get("passed")
            if isinstance(checks.get("checkpoint_loadable"), dict)
            else False,
            "class_mapping": record.get("actual_classes") == expected_classes,
            "dataset_available": dataset is not None
            and dataset.get("image_count", 0) > 0
            and dataset.get("label_file_count", 0) == dataset.get("image_count"),
            "dataset_identity": dataset_identity_matches,
            "evaluation_settings": evaluation_settings_valid,
            "evaluation_completed": record.get("validation_completed") is True,
            "metrics_present": (
                isinstance(summary, dict)
                and all(_number(summary.get(name)) for name in required_metrics)
                and isinstance(per_class, dict)
                and all(
                    isinstance(per_class.get(class_name), dict)
                    and all(
                        _number(per_class[class_name].get(metric))
                        for metric in ("precision", "recall", "map50", "map50_95")
                    )
                    for class_name in expected_classes
                )
                and isinstance(raw_metrics, dict)
                and bool(raw_metrics)
            ),
            "acceptance_thresholds": (
                len(metric_check_values) == len(threshold_names)
                and all(metric_check_values)
            ),
            "representative_tests": (
                isinstance(representative, dict)
                and representative.get("completed") is True
                and all(representative_checks.values())
                if isinstance(representative, dict)
                else False
            ),
        }
        for check_name, expected_passed in required_check_values.items():
            check = checks.get(check_name)
            if (
                isinstance(check, dict)
                and check.get("passed") is not expected_passed
            ):
                failures.append(
                    f"Required validation check {check_name} does not match its evidence."
                )

    if (
        record.get("validation_completed") is not True
        or record.get("validation_passed") is not True
    ):
        failures.append("Validation did not complete successfully and pass.")

    approved = record.get("production_alerts_approved") is True
    approval_time = _parse_timestamp(record.get("production_alerts_approved_at"))
    approval_valid = bool(
        approved
        and isinstance(record.get("production_alerts_approved_by"), str)
        and record["production_alerts_approved_by"].strip()
        and approval_time is not None
        and completed_at is not None
        and approval_time >= completed_at
        and approval_time <= current_time + timedelta(minutes=5)
        and record.get("production_alerts_approved_checkpoint_sha256")
        == record.get("checkpoint_sha256")
        and record.get("production_alerts_validation_sha256")
        == validation_report_sha256(record)
        and record.get("production_alerts_test_provider_accepted") is True
        and _parse_timestamp(
            record.get("production_alerts_test_provider_accepted_at")
        ) is not None
    )

    return {
        "passed": not failures,
        "failures": list(dict.fromkeys(failures)),
        "record": record,
        "checkpoint_path": str(checkpoint),
        "validation_record_path": str(sidecar),
        "production_alerts_approved": approval_valid and not failures,
        "production_alerts_approval_invalid": approved and not approval_valid,
        "validation_dataset": dataset,
    }
