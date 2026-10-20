"""Create a detection-format staging copy without changing source annotations."""

import math
import os
import shutil
from pathlib import Path

import yaml

from src.config import (
    EXPECTED_DETECTOR_CLASSES,
    YOLO_LABEL_BOUNDARY_TOLERANCE,
)


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "The_final-4"
DEST = ROOT / "The_final-4-detect"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}


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


def _format_number(value):
    return f"{value:.8f}"


def _convert_label_row(line, label_path, class_count):
    fields = line.split()
    if not fields:
        raise ValueError(f"{label_path}: empty annotation row.")
    try:
        class_id = int(fields[0])
    except ValueError as exc:
        raise ValueError(f"{label_path}: invalid class id {fields[0]!r}.") from exc
    if not 0 <= class_id < class_count:
        raise ValueError(
            f"{label_path}: Class id {class_id} is outside [0, {class_count - 1}]."
        )
    try:
        coordinates = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError(f"{label_path}: non-numeric annotation coordinate.") from exc
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
        raise ValueError(
            f"{label_path}: coordinates must be finite normalized values in [0, 1]."
        )

    converted = False
    if len(coordinates) == 4:
        center_x, center_y, width, height = coordinates
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        converted = True
    else:
        raise ValueError(
            f"{label_path}: expected four box coordinates or at least six "
            "even-count polygon coordinates."
        )
    if width <= 0 or height <= 0:
        raise ValueError(f"{label_path}: box width and height must be positive.")
    bounds = (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )
    overshoot = max(
        0.0,
        -bounds[0],
        -bounds[1],
        bounds[2] - 1,
        bounds[3] - 1,
    )
    if overshoot > YOLO_LABEL_BOUNDARY_TOLERANCE:
        raise ValueError(
            f"{label_path}: derived bounding box is outside the image."
        )
    clipped = overshoot > 0
    if clipped:
        x1, y1, x2, y2 = (
            max(0.0, bounds[0]),
            max(0.0, bounds[1]),
            min(1.0, bounds[2]),
            min(1.0, bounds[3]),
        )
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

    values = " ".join(_format_number(value) for value in (
        center_x,
        center_y,
        width,
        height,
    ))
    return f"{class_id} {values}", converted, clipped


def _resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def prepare_detection_dataset(source_yaml, staging_root):
    source_yaml = Path(source_yaml).expanduser().resolve()
    staging_root = Path(staging_root).expanduser().resolve()
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Missing dataset config: {source_yaml}")
    if not staging_root.is_dir():
        raise FileNotFoundError(
            f"Staging directory must exist and be empty: {staging_root}"
        )
    if any(staging_root.iterdir()):
        raise FileExistsError(
            f"Staging directory is not empty; refusing to overwrite: {staging_root}"
        )

    with source_yaml.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must be a YAML mapping.")
    class_names = _class_names(config)
    if (
        class_names != list(EXPECTED_DETECTOR_CLASSES)
        or config.get("nc") != len(EXPECTED_DETECTOR_CLASSES)
    ):
        raise ValueError("Dataset class mapping does not match the six-class contract.")

    source_root = _resolve(source_yaml.parent, config.get("path", source_yaml.parent))
    counts = {
        "linked_images": 0,
        "copied_images": 0,
        "box_rows": 0,
        "converted_polygon_rows": 0,
        "rounding_boundary_rows": 0,
        "empty_label_files": 0,
    }
    output_config = dict(config)
    output_config["path"] = str(staging_root)
    for canonical, aliases in SPLITS.items():
        source_value = next(
            (config[name] for name in aliases if name in config),
            None,
        )
        if source_value is None:
            raise ValueError(f"Dataset configuration has no {canonical} split.")
        values = source_value if isinstance(source_value, list) else [source_value]
        if len(values) != 1:
            raise ValueError(
                f"{canonical} split must contain exactly one image directory."
            )
        source_images = _resolve(source_root, values[0])
        source_labels = source_images.parent / "labels"
        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(
                f"Missing images or labels for {canonical}: {source_images}"
            )
        target_name = "valid" if canonical == "val" else canonical
        target_images = staging_root / target_name / "images"
        target_labels = staging_root / target_name / "labels"
        target_images.mkdir(parents=True)
        target_labels.mkdir(parents=True)

        image_paths = sorted(
            path for path in source_images.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        if not image_paths:
            raise ValueError(f"No images found in {canonical} split: {source_images}")
        for image_path in image_paths:
            relative_path = image_path.relative_to(source_images)
            staged_image = target_images / relative_path
            staged_image.parent.mkdir(parents=True, exist_ok=True)
            label_path = source_labels / relative_path.with_suffix(".txt")
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing label file for image {image_path}: {label_path}"
                )
            try:
                os.link(image_path, staged_image)
                counts["linked_images"] += 1
            except OSError:
                shutil.copy2(image_path, staged_image)
                counts["copied_images"] += 1

            staged_label = target_labels / relative_path.with_suffix(".txt")
            staged_label.parent.mkdir(parents=True, exist_ok=True)
            converted_rows = []
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8-sig").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    converted, was_polygon, clipped = _convert_label_row(
                        line,
                        label_path,
                        len(class_names),
                    )
                except ValueError as exc:
                    raise ValueError(f"{exc} Line {line_number}.") from exc
                converted_rows.append(converted)
                counts[
                    "converted_polygon_rows" if was_polygon else "box_rows"
                ] += 1
                if clipped:
                    counts["rounding_boundary_rows"] += 1
            if not converted_rows:
                counts["empty_label_files"] += 1
            staged_label.write_text(
                "\n".join(converted_rows)
                + ("\n" if converted_rows else ""),
                encoding="utf-8",
            )

    output_config["train"] = "train/images"
    output_config["val"] = "valid/images"
    output_config["test"] = "test/images"
    staged_yaml = staging_root / "data.yaml"
    staged_yaml.write_text(
        yaml.safe_dump(output_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return staged_yaml, counts


def main():
    if not (SOURCE / "data.yaml").is_file():
        raise FileNotFoundError(f"Missing dataset config: {SOURCE / 'data.yaml'}")
    if DEST.exists():
        raise FileExistsError(
            f"{DEST} already exists. Nothing was overwritten. "
            "Choose a new staging destination before running again."
        )
    DEST.mkdir(parents=True)
    staged_yaml, counts = prepare_detection_dataset(
        SOURCE / "data.yaml",
        DEST,
    )
    print(f"Prepared detection dataset: {staged_yaml}")
    print(f"Prepared annotation summary: {counts}")
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
