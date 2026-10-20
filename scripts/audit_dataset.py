"""Read-only integrity, class balance, duplicate, and visualization audit."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EXPECTED_DETECTOR_CLASSES,
    ROOT_DIR,
    VALIDATION_DATA_PATH,
    YOLO_LABEL_BOUNDARY_TOLERANCE,
)
from src.modeling.detectors.validation import IMAGE_SUFFIXES


SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}
COLORS = {
    0: (255, 160, 0),
    1: (255, 200, 0),
    2: (0, 0, 255),
    3: (255, 160, 100),
    4: (0, 80, 255),
    5: (0, 220, 100),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_names(configuration):
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


def _split_value(configuration, aliases):
    for alias in aliases:
        if alias in configuration:
            return configuration[alias]
    return None


def _resolve_path(root, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _split_directories(configuration, yaml_path):
    root_value = configuration.get("path", yaml_path.parent)
    root = _resolve_path(yaml_path.parent, root_value)
    directories = {}
    for split, aliases in SPLIT_ALIASES.items():
        value = _split_value(configuration, aliases)
        if isinstance(value, list):
            directories[split] = [_resolve_path(root, item) for item in value]
        elif isinstance(value, str):
            directories[split] = [_resolve_path(root, value)]
        else:
            directories[split] = []
    return root, directories


def _matching_label_path(image_path):
    parts = list(image_path.parts)
    image_index = next(
        (
            index for index in range(len(parts) - 1, -1, -1)
            if parts[index].lower() == "images"
        ),
        None,
    )
    if image_index is not None:
        parts[image_index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_yolo_label(label_path, class_count):
    """Parse detection boxes and polygons; report row errors without repair."""
    rows = []
    errors = []
    polygon_rows = 0
    box_rows = 0
    try:
        lines = Path(label_path).read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return rows, [f"Could not read label file: {exc}"], 0, 0

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
            if not 0 <= class_id < class_count:
                raise ValueError(f"class ID {class_id} is outside [0, {class_count - 1}]")
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("coordinates contain a non-finite value")
            if len(coordinates) == 4:
                x_center, y_center, width, height = coordinates
                if width <= 0 or height <= 0:
                    raise ValueError("box width and height must be positive")
                box = (
                    x_center - width / 2,
                    y_center - height / 2,
                    x_center + width / 2,
                    y_center + height / 2,
                )
                geometry = "box"
            elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                points = list(zip(coordinates[0::2], coordinates[1::2]))
                if len(points) < 3:
                    raise ValueError("polygon requires at least three points")
                box = (
                    min(point[0] for point in points),
                    min(point[1] for point in points),
                    max(point[0] for point in points),
                    max(point[1] for point in points),
                )
                geometry = "polygon"
            else:
                raise ValueError(
                    f"expected four box coordinates or an even polygon "
                    f"coordinate count, got {len(coordinates)}"
                )
            if any(value < 0 or value > 1 for value in coordinates):
                raise ValueError("normalized coordinates must be within [0, 1]")
            clipped = False
            if box[0] < 0 or box[1] < 0 or box[2] > 1 or box[3] > 1:
                overshoot = max(
                    0.0,
                    -box[0],
                    -box[1],
                    box[2] - 1,
                    box[3] - 1,
                )
                if overshoot > YOLO_LABEL_BOUNDARY_TOLERANCE:
                    raise ValueError("derived box extends outside the image")
                box = (
                    max(0.0, box[0]),
                    max(0.0, box[1]),
                    min(1.0, box[2]),
                    min(1.0, box[3]),
                )
                clipped = True
            if geometry == "box":
                box_rows += 1
            else:
                polygon_rows += 1
            rows.append({
                "line": line_number,
                "class_id": class_id,
                "geometry": geometry,
                "coordinates": coordinates,
                "box": box,
                "boundary_rounding_clipped": clipped,
            })
        except (ValueError, IndexError) as exc:
            errors.append(f"line {line_number}: {exc}")
    return rows, errors, box_rows, polygon_rows


def _render_sample(image_path, rows, class_names, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    for row in rows:
        class_id = row["class_id"]
        color = COLORS.get(class_id, (255, 255, 255))
        if row["geometry"] == "polygon":
            coords = row["coordinates"]
            points = np.asarray(
                [
                    [round(coords[index] * width), round(coords[index + 1] * height)]
                    for index in range(0, len(coords), 2)
                ],
                dtype=np.int32,
            )
            cv2.polylines(image, [points], True, color, 2)
        else:
            x1, y1, x2, y2 = row["box"]
            cv2.rectangle(
                image,
                (round(x1 * width), round(y1 * height)),
                (round(x2 * width), round(y2 * height)),
                color,
                2,
            )
        x1 = round(row["box"][0] * width)
        y1 = max(18, round(row["box"][1] * height) - 5)
        cv2.putText(
            image,
            class_names[class_id],
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write visualization: {output_path}")
    return True


def audit_dataset(data_yaml, output_directory=None, samples_per_class=2):
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8-sig") as file:
        configuration = yaml.safe_load(file)
    if not isinstance(configuration, dict):
        raise ValueError("Dataset configuration must be a YAML mapping.")

    class_names = _class_names(configuration)
    root, split_directories = _split_directories(configuration, yaml_path)
    expected = list(EXPECTED_DETECTOR_CLASSES)
    configured_nc = configuration.get("nc")
    class_mapping_valid = (
        class_names == expected
        and configured_nc == len(expected)
    )
    report_directory = (
        Path(output_directory).resolve()
        if output_directory is not None
        else Path(ROOT_DIR) / "runs" / "dataset_audit"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    visualization_directory = report_directory / "visualizations"
    split_paths = {}
    for split, directories in split_directories.items():
        files = []
        for directory in directories:
            if directory.is_dir():
                files.extend(
                    path for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
        split_paths[split] = sorted(set(files))

    class_counts = {
        split: [0] * len(expected)
        for split in SPLIT_ALIASES
    }
    class_image_counts = {
        split: [0] * len(expected)
        for split in SPLIT_ALIASES
    }
    split_report = {}
    hash_locations = defaultdict(list)
    basename_locations = defaultdict(list)
    visualization_paths = []

    for split, image_paths in split_paths.items():
        split_label_dirs = [
            _resolve_path(path.parent, "labels")
            if path.name.lower() == "images" else path.parent / "labels"
            for path in split_directories[split]
        ]
        split_stats = {
            "image_count": len(image_paths),
            "readable_images": 0,
            "corrupt_images": [],
            "label_files_found": 0,
            "valid_label_files": 0,
            "missing_label_files": [],
            "empty_label_files": [],
            "extra_label_files": [],
            "invalid_annotation_files": [],
            "invalid_annotation_rows": 0,
            "valid_annotation_rows": 0,
            "rounding_boundary_rows": 0,
            "bounding_box_rows": 0,
            "polygon_rows": 0,
            "classes": {},
        }
        image_by_label = {}
        for image_path in image_paths:
            label_path = _matching_label_path(image_path)
            image_by_label[label_path.resolve()] = image_path
            basename_locations[image_path.stem.lower()].append({
                "split": split,
                "path": str(image_path),
            })
            try:
                digest = sha256_file(image_path)
                hash_locations[digest].append({
                    "split": split,
                    "path": str(image_path),
                })
            except OSError as exc:
                split_stats["corrupt_images"].append({
                    "path": str(image_path),
                    "error": f"Image bytes could not be read: {exc}",
                })
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                split_stats["corrupt_images"].append({
                    "path": str(image_path),
                    "error": "OpenCV could not decode the image.",
                })
                continue
            split_stats["readable_images"] += 1
            if not label_path.is_file():
                split_stats["missing_label_files"].append(str(label_path))
                continue
            split_stats["label_files_found"] += 1
            rows, errors, box_rows, polygon_rows = parse_yolo_label(
                label_path,
                len(expected),
            )
            if not errors:
                split_stats["valid_label_files"] += 1
            split_stats["valid_annotation_rows"] += len(rows)
            split_stats["bounding_box_rows"] += box_rows
            split_stats["polygon_rows"] += polygon_rows
            split_stats["invalid_annotation_rows"] += len(errors)
            if errors:
                split_stats["invalid_annotation_files"].append({
                    "path": str(label_path),
                    "errors": errors,
                })
            if not rows:
                split_stats["empty_label_files"].append(str(label_path))
            present_classes = set()
            for row in rows:
                class_id = row["class_id"]
                if row.get("boundary_rounding_clipped"):
                    split_stats["rounding_boundary_rows"] += 1
                class_counts[split][class_id] += 1
                present_classes.add(class_id)
                if (
                    len([
                        path for path in visualization_paths
                        if path.get("class") == expected[class_id]
                    ]) < samples_per_class
                ):
                    output_path = (
                        visualization_directory / split /
                        f"{class_id}-{image_path.stem}.jpg"
                    )
                    if _render_sample(
                        image_path,
                        rows,
                        expected,
                        output_path,
                    ):
                        visualization_paths.append({
                            "class": expected[class_id],
                            "split": split,
                            "source": str(image_path),
                            "visualization": str(output_path),
                        })

            for class_id in present_classes:
                class_image_counts[split][class_id] += 1

        for label_dir in split_label_dirs:
            if not label_dir.is_dir():
                continue
            for label_path in label_dir.rglob("*.txt"):
                if label_path.resolve() not in image_by_label:
                    split_stats["extra_label_files"].append(str(label_path))

        split_stats["classes"] = {
            name: {
                "instances": class_counts[split][class_id],
                "images": class_image_counts[split][class_id],
            }
            for class_id, name in enumerate(expected)
        }
        split_report[split] = split_stats

    duplicate_groups = [
        locations for locations in hash_locations.values()
        if len(locations) > 1
    ]
    exact_leakage_groups = [
        locations for locations in duplicate_groups
        if len({location["split"] for location in locations}) > 1
    ]
    suspected_basename_groups = [
        locations for locations in basename_locations.values()
        if len(locations) > 1
        and len({location["split"] for location in locations}) > 1
    ]

    all_counts = {
        name: sum(class_counts[split][class_id] for split in SPLIT_ALIASES)
        for class_id, name in enumerate(expected)
    }
    max_count = max(all_counts.values(), default=0)
    imbalance = {
        name: (round(max_count / count, 3) if count else None)
        for name, count in all_counts.items()
    }
    errors_exist = any(
        split_report[split]["corrupt_images"]
        or split_report[split]["missing_label_files"]
        or split_report[split]["invalid_annotation_rows"]
        or split_report[split]["extra_label_files"]
        for split in split_report
    )
    report = {
        "schema_version": 1,
        "audit_completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_yaml": str(yaml_path),
        "dataset_yaml_sha256": sha256_file(yaml_path),
        "dataset_root": str(root),
        "dataset_path_exists": root.is_dir(),
        "configured_nc": configured_nc,
        "configured_classes": class_names,
        "expected_classes": expected,
        "class_mapping_valid": class_mapping_valid,
        "path_verification": {
            split: [
                {"path": str(path), "exists": path.is_dir()}
                for path in directories
            ]
            for split, directories in split_directories.items()
        },
        "splits": split_report,
        "class_balance": {
            name: {
                "instances": count,
                "relative_to_largest_class": imbalance[name],
            }
            for name, count in all_counts.items()
        },
        "duplicates": {
            "exact_duplicate_groups": duplicate_groups,
            "exact_cross_split_leakage": exact_leakage_groups,
            "same_basename_cross_split_suspicions": suspected_basename_groups,
        },
        "annotation_geometry": {
            "types_found": sorted({
                "bbox"
                for split in split_report.values()
                if split["bounding_box_rows"]
            } | {
                "polygon"
                for split in split_report.values()
                if split["polygon_rows"]
            }),
            "source_annotations_modified": False,
            "training_staging_normalization": (
                "Bounding boxes overshooting an image boundary by at most "
                f"{YOLO_LABEL_BOUNDARY_TOLERANCE:g} are clipped to [0, 1] "
                "only in the training staging copy. The source labels remain "
                "unchanged."
            ),
            "boundary_rounding_tolerance": YOLO_LABEL_BOUNDARY_TOLERANCE,
            "note": (
                "Training preparation converts valid polygon annotations to "
                "enclosing detection boxes in a separate staging copy."
            ),
        },
        "visualizations": visualization_paths,
        "audit_passed": (
            class_mapping_valid
            and not errors_exist
            and not exact_leakage_groups
            and all(
                split_report[split]["image_count"] > 0
                for split in ("train", "val", "test")
            )
        ),
        "warnings": [],
    }
    if suspected_basename_groups:
        report["warnings"].append(
            "Matching filenames occur across splits; inspect the reported "
            "files for near-duplicate leakage."
        )
    rounding_rows = sum(
        values["rounding_boundary_rows"]
        for values in split_report.values()
    )
    if rounding_rows:
        report["warnings"].append(
            f"{rounding_rows} bounding boxes exceed an image edge by no more "
            "than the six-decimal annotation rounding tolerance (one "
            "millionth of the normalized image dimension). The source labels "
            "are unchanged; training staging clips only this tiny overshoot."
        )
    if max_count and min(all_counts.values(), default=0) < max_count * 0.05:
        report["warnings"].append(
            "At least one class has fewer than 5% of the largest class's "
            "instances; review class-specific examples and performance."
        )

    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / "dataset_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(VALIDATION_DATA_PATH),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.samples_per_class < 0:
        parser.error("--samples-per-class cannot be negative.")
    report, report_path = audit_dataset(
        arguments.data,
        arguments.output,
        arguments.samples_per_class,
    )
    print(f"Dataset audit report: {report_path}")
    print(f"Dataset audit passed: {report['audit_passed']}")
    for split, values in report["splits"].items():
        print(
            f"{split}: {values['image_count']} images, "
            f"{values['label_files_found']} labels, "
            f"{values['invalid_annotation_rows']} invalid annotation rows"
        )
    print("Class instances:")
    for name, values in report["class_balance"].items():
        print(f"  {name}: {values['instances']}")
    print(
        "Exact cross-split duplicate groups:",
        len(report["duplicates"]["exact_cross_split_leakage"]),
    )
    print("Warnings:", json.dumps(report["warnings"]))
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
