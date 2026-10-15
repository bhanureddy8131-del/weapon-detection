"""Create an isolated YOLO-detection dataset from mixed box/polygon labels."""

import math
import os
from datetime import datetime
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _convert_label_row(row, label_path, class_count):
    values = row.split()
    if not values:
        return None, False

    try:
        class_id = int(values[0])
        coordinates = [float(value) for value in values[1:]]
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value in {label_path}: {row!r}"
        ) from exc

    if not 0 <= class_id < class_count:
        raise ValueError(
            f"Class id {class_id} is outside [0, {class_count}) in "
            f"{label_path}."
        )
    if any(
        not math.isfinite(value) or value < -1e-6 or value > 1.000001
        for value in coordinates
    ):
        raise ValueError(
            f"Coordinates must be finite normalized values in {label_path}: "
            f"{row!r}"
        )

    if len(coordinates) == 4:
        center_x, center_y, width, height = coordinates
        converted = False
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        points = list(zip(coordinates[::2], coordinates[1::2]))
        x_min = min(point[0] for point in points)
        x_max = max(point[0] for point in points)
        y_min = min(point[1] for point in points)
        y_max = max(point[1] for point in points)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        converted = True
    else:
        raise ValueError(
            f"Expected a YOLO box or polygon row in {label_path}: {row!r}"
        )

    if width <= 0 or height <= 0:
        raise ValueError(f"Degenerate bounding box in {label_path}: {row!r}")

    return (
        f"{class_id} {center_x:.8f} {center_y:.8f} "
        f"{width:.8f} {height:.8f}",
        converted,
    )


def prepare_detection_dataset(source_yaml, destination_root):
    """Hard-link source images and write converted labels into a new tree."""
    source_yaml = Path(source_yaml).resolve()
    destination_root = Path(destination_root).resolve()
    with source_yaml.open("r", encoding="utf-8-sig") as config_file:
        cfg = yaml.safe_load(config_file)

    source_root = Path(cfg["path"])
    if not source_root.is_absolute():
        source_root = source_yaml.parent / source_root
    class_count = int(cfg["nc"])
    counts = {
        "linked_images": 0,
        "converted_polygon_rows": 0,
        "box_rows": 0,
        "empty_labels": 0,
    }
    splits = {}

    for split_name, config_key in (
        ("train", "train"),
        ("valid", "val"),
        ("test", "test"),
    ):
        relative_images = Path(cfg[config_key])
        source_images = source_root / relative_images
        source_labels = source_images.parent / "labels"
        if not source_images.is_dir():
            raise FileNotFoundError(
                f"{split_name} images directory is missing: {source_images}"
            )
        if not source_labels.is_dir():
            raise FileNotFoundError(
                f"{split_name} labels directory is missing: {source_labels}"
            )

        target_images = destination_root / split_name / "images"
        target_labels = destination_root / split_name / "labels"
        target_images.mkdir(parents=True, exist_ok=False)
        target_labels.mkdir(parents=True, exist_ok=False)

        for source_image in source_images.iterdir():
            if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target_image = target_images / source_image.name
            try:
                os.link(source_image, target_image)
            except OSError as exc:
                raise OSError(
                    "Could not hard-link dataset image into the staging "
                    f"directory: {source_image} -> {target_image}. "
                    "The source dataset was left unchanged."
                ) from exc
            counts["linked_images"] += 1

            source_label = source_labels / f"{source_image.stem}.txt"
            if not source_label.is_file():
                continue
            try:
                rows = source_label.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as exc:
                raise RuntimeError(
                    f"Could not read annotation file {source_label}: {exc}"
                ) from exc

            converted_rows = []
            for row in rows:
                converted_row, was_polygon = _convert_label_row(
                    row,
                    source_label,
                    class_count,
                )
                if converted_row is None:
                    continue
                converted_rows.append(converted_row)
                if was_polygon:
                    counts["converted_polygon_rows"] += 1
                else:
                    counts["box_rows"] += 1

            if not converted_rows:
                counts["empty_labels"] += 1
            target_label = target_labels / source_label.name
            target_label.write_text(
                "\n".join(converted_rows) + ("\n" if converted_rows else ""),
                encoding="utf-8",
            )

        splits[config_key] = f"{split_name}/images"

    staged_cfg = dict(cfg)
    staged_cfg["path"] = str(destination_root)
    staged_cfg.update(splits)
    staged_yaml = destination_root / "data.yaml"
    staged_yaml.write_text(
        yaml.safe_dump(staged_cfg, sort_keys=False),
        encoding="utf-8",
    )
    return staged_yaml, counts


def create_staging_directory(parent_directory):
    parent_directory = Path(parent_directory)
    parent_directory.mkdir(parents=True, exist_ok=True)
    name = f"prepared_detection_dataset_{datetime.now():%Y%m%d_%H%M%S}"
    staging_root = parent_directory / name
    staging_root.mkdir(parents=True, exist_ok=False)
    return staging_root
