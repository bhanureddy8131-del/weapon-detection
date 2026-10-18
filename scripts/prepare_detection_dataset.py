
from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "The_final-4"
DEST = ROOT / "The_final-4-detect"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def polygon_to_box(values):
    """Convert normalized polygon x,y pairs into YOLO x_center,y_center,w,h."""
    xs = values[0::2]
    ys = values[1::2]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    return [
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        x2 - x1,
        y2 - y1,
    ]


def format_number(value):
    return f"{value:.8f}".rstrip("0").rstrip(".")


def main():
    if not (SOURCE / "data.yaml").is_file():
        raise FileNotFoundError(f"Missing dataset config: {SOURCE / 'data.yaml'}")

    if DEST.exists():
        raise FileExistsError(
            f"{DEST} already exists. Nothing was overwritten. "
            "Inspect it or choose a new destination before running again."
        )

    config = yaml.safe_load((SOURCE / "data.yaml").read_text(encoding="utf-8"))

    # Keep the expected classes explicit and verify source config.
    names = config.get("names")
    print("Dataset classes:", names)

    if not names or len(names) != 6:
        raise ValueError("Expected exactly six classes in data.yaml.")

    DEST.mkdir(parents=True)

    stats = {
        "box_rows": 0,
        "polygon_rows_converted": 0,
        "empty_files": 0,
        "bad_rows": 0,
    }

    for split in ("train", "valid", "test"):
        source_images = SOURCE / split / "images"
        source_labels = SOURCE / split / "labels"
        dest_images = DEST / split / "images"
        dest_labels = DEST / split / "labels"

        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"Missing images or labels for split: {split}")

        dest_images.mkdir(parents=True)
        dest_labels.mkdir(parents=True)

        print(f"\nPreparing {split}...")

        # Copy image files and preserve relative filenames.
        for image_path in source_images.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                shutil.copy2(image_path, dest_images / image_path.name)

        # Copy and normalize annotations into detection-only YOLO labels.
        for label_path in source_labels.glob("*.txt"):
            output_rows = []

            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                try:
                    class_id = int(parts[0])
                    values = [float(v) for v in parts[1:]]

                    if class_id < 0 or class_id >= len(names):
                        raise ValueError("class ID outside configured class range")

                    if len(values) == 4:
                        # Standard YOLO detection box.
                        x, y, w, h = values
                        stats["box_rows"] += 1

                    elif len(values) >= 6 and len(values) % 2 == 0:
                        # Polygon coordinates: x1,y1,x2,y2,...
                        x, y, w, h = polygon_to_box(values)
                        stats["polygon_rows_converted"] += 1

                    else:
                        raise ValueError(
                            f"Unexpected number of coordinate values: {len(values)}"
                        )

                    # Ensure valid normalized coordinates.
                    if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                        raise ValueError("coordinates outside normalized range")
                    if w <= 0 or h <= 0:
                        raise ValueError("non-positive box width or height")

                    output_rows.append(
                        f"{class_id} " + " ".join(
                            format_number(v) for v in (x, y, w, h)
                        )
                    )

                except Exception as exc:
                    stats["bad_rows"] += 1
                    raise ValueError(
                        f"{label_path}, line {line_number}: {exc}"
                    ) from exc

            if not output_rows:
                stats["empty_files"] += 1

            (dest_labels / label_path.name).write_text(
                "\n".join(output_rows) + ("\n" if output_rows else ""),
                encoding="utf-8",
            )

        print(f"Finished {split}")

    # Build a dataset config pointing at the new copy.
    config["path"] = str(DEST).replace("\\", "/")
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    output_yaml = DEST / "data.yaml"
    output_yaml.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("\nDataset preparation complete.")
    print("New dataset:", DEST)
    print("Config:", output_yaml)
    print("Original dataset was not modified.")
    print("Standard box rows preserved:", stats["box_rows"])
    print("Polygon rows converted:", stats["polygon_rows_converted"])
    print("Empty label files preserved:", stats["empty_files"])
    print("Invalid rows:", stats["bad_rows"])


if __name__ == "__main__":
    main()
