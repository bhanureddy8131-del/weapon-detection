import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Results

if __package__:
    from .prepare_detection_dataset import (
        create_staging_directory,
        prepare_detection_dataset,
    )
else:
    from prepare_detection_dataset import (
        create_staging_directory,
        prepare_detection_dataset,
    )

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "The_final-4" / "data.yaml"
BASE_MODEL = ROOT / "yolo11s.pt"
OUTPUT_DIR = ROOT / "SmartVision_Final"
PREPARED_DATA_ROOT = OUTPUT_DIR / "prepared_datasets"
MODEL_DEST = ROOT / "models" / "yolo11s_fine_tune_v3.pt"
VALIDATION_DEST = ROOT / "models" / "yolo11s_fine_tune_v3.pt.validation.json"


def get_representative_validation_images(dataset_root, cfg):
    image_dir = dataset_root / cfg["val"]
    label_dir = image_dir.parent / "labels"
    classes_to_find = set(range(len(cfg["names"])))
    representative_images = {}
    image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for label_path in sorted(label_dir.glob("*.txt")):
        try:
            label_rows = label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                f"Could not read validation label file {label_path}: {exc}"
            ) from exc

        row_classes = set()
        for row in label_rows:
            parts = row.split()
            if parts:
                try:
                    row_classes.add(int(parts[0]))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid YOLO class id in {label_path}: {row!r}"
                    ) from exc

        for class_id in classes_to_find.intersection(row_classes):
            for suffix in image_suffixes:
                image_path = image_dir / f"{label_path.stem}{suffix}"
                if image_path.is_file():
                    representative_images[class_id] = image_path
                    break
        classes_to_find.difference_update(representative_images)
        if not classes_to_find:
            break

    return representative_images


def main():
    print("=" * 60)
    print("SMARTVISION PRO - YOLO CPU TRAINING")
    print("=" * 60)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset config missing: {DATA_YAML}")

    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Base model missing: {BASE_MODEL}")

    with DATA_YAML.open("r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    staging_root = create_staging_directory(PREPARED_DATA_ROOT)
    training_data_yaml, preparation_counts = prepare_detection_dataset(
        DATA_YAML,
        staging_root,
    )

    print("Dataset:", DATA_YAML)
    print("Prepared detection dataset:", training_data_yaml)
    print("Prepared annotation summary:", preparation_counts)
    print("Classes:", cfg["names"])
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    dataset_root = Path(cfg["path"])
    for split in ("train", "val", "test"):
        image_dir = dataset_root / cfg[split]
        if not image_dir.is_dir():
            raise FileNotFoundError(f"{split} images missing: {image_dir}")

    print("\nLoading pretrained YOLO11s model...")
    model = YOLO(str(BASE_MODEL))

    run_name = f"weapon_cpu_trial_{datetime.now():%Y%m%d_%H%M%S}"
    print("\nStarting a 10-epoch CPU trial.")
    print("Training output:", OUTPUT_DIR / run_name)
    model.train(
        data=str(training_data_yaml),
        epochs=10,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=2,
        patience=5,
        cache=False,
        amp=False,
        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=False,
        plots=True,
        save=True,
        verbose=True,
    )

    run_dir = OUTPUT_DIR / run_name
    best_weights = run_dir / "weights" / "best.pt"

    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Training finished, but best.pt was not found at {best_weights}"
        )

    print("\nValidating the best checkpoint on the held-out validation split...")
    validated_model = YOLO(str(best_weights))
    trained_names = tuple(
        str(name).strip().lower()
        for _, name in sorted(validated_model.names.items())
    )
    expected_names = tuple(str(name).lower() for name in cfg["names"].values())
    if trained_names != expected_names:
        raise ValueError(
            "Checkpoint class mapping does not match data.yaml: "
            f"{trained_names} != {expected_names}"
        )

    validation_results = validated_model.val(
        data=str(training_data_yaml),
        split="val",
        imgsz=416,
        batch=4,
        device="cpu",
        workers=2,
        plots=False,
    )
    representative_images = get_representative_validation_images(
        staging_root,
        cfg,
    )
    expected_class_ids = set(range(len(expected_names)))
    representative_validation_completed = (
        set(representative_images) == expected_class_ids
    )
    if not representative_images:
        print(
            "Warning: no per-class validation images were located. "
            "Production alerts will remain disabled."
        )
    elif not representative_validation_completed:
        missing = sorted(expected_class_ids - set(representative_images))
        print(
            "Warning: representative validation images are missing for "
            f"class IDs {missing}. Production alerts will remain disabled."
        )
    else:
        representative_dir = run_dir / "representative_validation"
        validation_predictions = validated_model.predict(
            source=[str(path) for path in representative_images.values()],
            imgsz=416,
            device="cpu",
            save=True,
            project=str(representative_dir.parent),
            name=representative_dir.name,
            exist_ok=False,
            verbose=False,
        )
        print("Representative validation predictions:")
        for image_result in validation_predictions:
            if isinstance(image_result, Results):
                boxes = image_result.boxes
                print(
                    " ",
                    Path(image_result.path).name,
                    "detections:",
                    len(boxes) if boxes is not None else 0,
                )

    digest = hashlib.sha256()
    with best_weights.open("rb") as weights_file:
        for chunk in iter(lambda: weights_file.read(1024 * 1024), b""):
            digest.update(chunk)

    MODEL_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, MODEL_DEST)
    validation_metrics = {}
    for key, value in validation_results.results_dict.items():
        try:
            validation_metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    metric_summary = {}
    for target, search_term in (
        ("precision", "precision"),
        ("recall", "recall"),
        ("map50", "map50("),
        ("map50_95", "map50-95"),
    ):
        matching_values = [
            value
            for key, value in validation_metrics.items()
            if search_term in key.lower()
        ]
        if not matching_values:
            raise ValueError(
                f"Validation did not report the required {target} metric."
            )
        metric_summary[target] = matching_values[0]

    validation_record = {
        "validation_completed": True,
        "weights_sha256": digest.hexdigest(),
        "classes": list(expected_names),
        "dataset_config": str(training_data_yaml),
        "validation_metrics": validation_metrics,
        "validation_summary": metric_summary,
        "representative_validation_completed": (
            representative_validation_completed
        ),
        "representative_validation_images": {
            expected_names[class_id]: str(image_path)
            for class_id, image_path in representative_images.items()
        },
        "production_alerts_approved": False,
    }
    VALIDATION_DEST.write_text(
        json.dumps(validation_record, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("TRAINING TRIAL COMPLETED")
    print("Best checkpoint:", best_weights)
    print("Application model:", MODEL_DEST)
    print("Validation record:", VALIDATION_DEST)
    print("Validation metrics:", validation_metrics)
    print(
        "Representative validation complete:",
        representative_validation_completed,
    )
    print("Production alerts approved: False (operator review required)")
    print("Training results:", run_dir)
    print("=" * 60)

if __name__ == "__main__":
    main()
