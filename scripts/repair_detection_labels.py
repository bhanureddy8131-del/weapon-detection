from pathlib import Path
from datetime import datetime
import math
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "The_final-4"
LABEL_DIRS = [
    DATASET / "train" / "labels",
    DATASET / "valid" / "labels",
    DATASET / "test" / "labels",
]
CLASS_IDS = set(range(6))
EPSILON = 1e-6


def parse_label_file(path):
    original = path.read_text(encoding="utf-8-sig")
    output = []
    polygon_count = 0

    for line_number, raw_line in enumerate(original.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        try:
            values = [float(item) for item in parts]
        except ValueError:
            raise ValueError(
                f"{path}:{line_number}: non-numeric annotation"
            )

        if not all(math.isfinite(v) for v in values):
            raise ValueError(
                f"{path}:{line_number}: non-finite number"
            )

        class_value = values[0]
        if class_value != int(class_value) or int(class_value) not in CLASS_IDS:
            raise ValueError(
                f"{path}:{line_number}: invalid class ID {class_value}"
            )

        class_id = int(class_value)

        if len(values) == 5:
            # Existing YOLO detection box: class, cx, cy, width, height.
            _, cx, cy, width, height = values
            if not (
                0 <= cx <= 1 and 0 <= cy <= 1
                and 0 < width <= 1 and 0 < height <= 1
                and cx - width / 2 >= -EPSILON
                and cy - height / 2 >= -EPSILON
                and cx + width / 2 <= 1 + EPSILON
                and cy + height / 2 <= 1 + EPSILON
            ):
                raise ValueError(
                    f"{path}:{line_number}: invalid normalized box"
                )
            output.append(
                f"{class_id} {cx:.6f} {cy:.6f} "
                f"{width:.6f} {height:.6f}"
            )

        elif len(values) >= 7 and len(values) % 2 == 1:
            # YOLO segmentation polygon: class, x1, y1, x2, y2, ...
            coords = values[1:]
            if len(coords) < 6:
                raise ValueError(
                    f"{path}:{line_number}: polygon has too few coordinates"
                )
            if any(coord < 0 or coord > 1 for coord in coords):
                raise ValueError(
                    f"{path}:{line_number}: polygon coordinate outside [0, 1]"
                )

            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"{path}:{line_number}: polygon has zero area"
                )

            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            output.append(
                f"{class_id} {cx:.6f} {cy:.6f} "
                f"{width:.6f} {height:.6f}"
            )
            polygon_count += 1

        else:
            raise ValueError(
                f"{path}:{line_number}: unsupported row with "
                f"{len(values)} values"
            )

    return original, "\n".join(output) + ("\n" if output else ""), polygon_count


def main():
    missing = [str(p) for p in LABEL_DIRS if not p.is_dir()]
    if missing:
        print("ERROR: Missing label directories:")
        print("\n".join(missing))
        return 1

    files = []
    for folder in LABEL_DIRS:
        files.extend(sorted(folder.glob("*.txt")))

    if not files:
        print("ERROR: No label files found.")
        return 1

    changes = []
    errors = []
    polygon_total = 0

    print(f"Checking {len(files)} label files. No files changed yet.")

    for path in files:
        try:
            original, repaired, polygon_count = parse_label_file(path)
            polygon_total += polygon_count
            if original.strip() != repaired.strip():
                changes.append((path, original, repaired))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    if errors:
        print("\nERRORS FOUND. Nothing was modified:")
        for error in errors[:100]:
            print(error)
        if len(errors) > 100:
            print(f"... and {len(errors) - 100} more errors.")
        return 1

    print(f"Polygon rows to convert: {polygon_total}")
    print(f"Files requiring changes: {len(changes)}")

    if not changes:
        print("No changes needed.")
        return 0

    backup_root = ROOT / "label_backups" / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    backup_root.mkdir(parents=True, exist_ok=False)

    # Back up and verify every file before modifying any original.
    try:
        for path, original, _ in changes:
            relative = path.relative_to(DATASET)
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            if backup.read_text(encoding="utf-8-sig") != original:
                raise OSError(f"Backup verification failed: {backup}")
    except OSError as exc:
        print(f"ERROR creating verified backups: {exc}")
        print("Original label files have not been modified.")
        print(f"Backup folder: {backup_root}")
        return 1

    # Write repaired contents using temporary files first.
    temporary_files = []
    try:
        for path, _, repaired in changes:
            temp = path.with_name(path.name + ".repair_tmp")
            temp.write_text(repaired, encoding="utf-8", newline="\n")
            temporary_files.append((temp, path))

        for temp, path in temporary_files:
            temp.replace(path)

    except OSError as exc:
        print(f"ERROR while writing repaired labels: {exc}")
        print("Restore originals from the backup folder if needed:")
        print(backup_root)
        return 1

    print("\nRepair completed.")
    print(f"Verified backups: {backup_root}")
    print(f"Modified files: {len(changes)}")
    print("Polygon annotations were converted to rectangular boxes.")
    print("Run scripts/find_mixed_labels.py to verify the result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())