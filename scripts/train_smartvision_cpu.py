
"""
SmartVision Pro - YOLO11 CPU Training

Trains the six-class SmartVision object detector using the
existing dataset and pretrained YOLO11 small checkpoint.

IMPORTANT:
- This script trains and evaluates a model.
- It does not approve the model for production alerts.
- It does not modify the live detector configuration.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "The_final-4" / "data.yaml"
BASE_MODEL = ROOT / "yolo11s.pt"

OUTPUT_DIR = ROOT / "runs" / "detect"
RUN_NAME = "smartvision_cpu_retrain"

EXPECTED_CLASSES = [
    "atm",
    "backpack",
    "gun",
    "handbag",
    "knife",
    "person",
]

# Initial training configuration.
# CPU training can be slow; evaluate the first run before
# committing to a longer training schedule.
EPOCHS = 30
IMAGE_SIZE = 416
BATCH_SIZE = 4
WORKERS = 0
PATIENCE = 10
SEED = 42

DEVICE = "cpu"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("SmartVisionTraining")


# ============================================================
# DATASET VALIDATION
# ============================================================

def load_dataset_config():
    """Check YAML syntax, class mapping, and dataset directories."""

    if not DATA.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA}"
        )

    if not BASE_MODEL.is_file():
        raise FileNotFoundError(
            f"Pretrained YOLO checkpoint not found: {BASE_MODEL}"
        )

    with DATA.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping.")

    names = config.get("names")

    if isinstance(names, dict):
        normalized_names = [
            str(names[key])
            for key in sorted(names, key=lambda value: int(value))
        ]
    elif isinstance(names, list):
        normalized_names = [str(name) for name in names]
    else:
        raise ValueError(
            "Dataset YAML must define class names as a list or mapping."
        )

    normalized_names = [
        name.strip().lower() for name in normalized_names
    ]

    if normalized_names != EXPECTED_CLASSES:
        raise ValueError(
            "Dataset class mapping does not match SmartVision.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found:    {normalized_names}"
        )

    if int(config.get("nc", -1)) != len(EXPECTED_CLASSES):
        raise ValueError(
            f"Expected nc={len(EXPECTED_CLASSES)}, "
            f"found nc={config.get('nc')}"
        )

    dataset_root = Path(config.get("path", ROOT / "The_final-4"))

    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root

    dataset_root = dataset_root.resolve()

    for split in ("train", "val"):
        relative_path = config.get(split)

        if not relative_path:
            raise ValueError(
                f"Dataset YAML is missing the '{split}' path."
            )

        split_path = Path(relative_path)

        if not split_path.is_absolute():
            split_path = dataset_root / split_path

        split_path = split_path.resolve()

        if not split_path.is_dir():
            raise FileNotFoundError(
                f"{split} images directory does not exist: "
                f"{split_path}"
            )

        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".tif",
            ".tiff",
        }

        image_count = sum(
            1
            for path in split_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in supported_extensions
        )

        if image_count == 0:
            raise ValueError(
                f"No supported images found in {split_path}"
            )

        logger.info(
            "%s images directory: %s",
            split,
            split_path,
        )
        logger.info("%s image count: %s", split, image_count)

    logger.info("Dataset configuration: %s", DATA)
    logger.info("Pretrained checkpoint: %s", BASE_MODEL)
    logger.info("Dataset classes: %s", normalized_names)

    return normalized_names


# ============================================================
# METRIC SERIALIZATION
# ============================================================

def normalize_metric(value):
    """Convert NumPy and other scalar values to JSON-safe values."""

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, (int, float, str, bool)) or value is None:
        return value

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def collect_metrics(metrics):
    """Extract measured detection metrics without inventing values."""

    results = getattr(metrics, "results_dict", {})

    summary = {}

    for key, value in results.items():
        summary[str(key)] = normalize_metric(value)

    box_metrics = getattr(metrics, "box", None)

    if box_metrics is not None:
        summary["precision"] = normalize_metric(
            getattr(box_metrics, "mp", None)
        )
        summary["recall"] = normalize_metric(
            getattr(box_metrics, "mr", None)
        )
        summary["map50"] = normalize_metric(
            getattr(box_metrics, "map50", None)
        )
        summary["map50_95"] = normalize_metric(
            getattr(box_metrics, "map", None)
        )

    return summary


# ============================================================
# TRAINING
# ============================================================

def main():
    start_time = time.monotonic()

    logger.info("=" * 65)
    logger.info("SMARTVISION PRO - YOLO TRAINING")
    logger.info("=" * 65)

    class_names = load_dataset_config()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", DEVICE)
    logger.info("Epochs: %s", EPOCHS)
    logger.info("Image size: %s", IMAGE_SIZE)
    logger.info("Batch size: %s", BATCH_SIZE)
    logger.info("Workers: %s", WORKERS)

    logger.warning(
        "CPU training may take a considerable amount of time."
    )

    logger.warning(
        "The resulting model will NOT automatically be approved "
        "for production alerts."
    )

    # Load the pretrained detector.
    model = YOLO(str(BASE_MODEL))

    # Train using the existing six-class dataset.
    train_results = model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=False,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        cache=False,
        amp=False,
        close_mosaic=10,
        verbose=True,
    )

    # Use the actual run directory returned by Ultralytics.
    save_dir = Path(train_results.save_dir).resolve()

    best_checkpoint = save_dir / "weights" / "best.pt"
    last_checkpoint = save_dir / "weights" / "last.pt"

    logger.info("Training directory: %s", save_dir)
    logger.info("Best checkpoint: %s", best_checkpoint)
    logger.info("Last checkpoint: %s", last_checkpoint)

    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            "Training finished, but best.pt was not created. "
            f"Expected checkpoint: {best_checkpoint}"
        )

    # Load the best checkpoint and run an explicit validation pass.
    logger.info("=" * 65)
    logger.info("VALIDATING THE BEST CHECKPOINT")
    logger.info("=" * 65)

    best_model = YOLO(str(best_checkpoint))

    validation_metrics = best_model.val(
        data=str(DATA),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        project=str(OUTPUT_DIR),
        name=f"{save_dir.name}_validation",
        exist_ok=True,
        verbose=True,
    )

    measured_metrics = collect_metrics(validation_metrics)

    summary = {
        "project": "SmartVision Pro",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_yaml": str(DATA),
        "base_checkpoint": str(BASE_MODEL),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": (
            str(last_checkpoint)
            if last_checkpoint.is_file()
            else None
        ),
        "training_directory": str(save_dir),
        "device": DEVICE,
        "epochs_requested": EPOCHS,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "class_names": class_names,
        "validation_metrics": measured_metrics,
        "training_duration_seconds": round(
            time.monotonic() - start_time,
            2,
        ),
        "production_alerts_approved": False,
        "production_note": (
            "This training report is not a production approval. "
            "Run the project's required detector acceptance tests "
            "and validation workflow before using the checkpoint "
            "for automatic security notifications."
        ),
    }

    summary_path = save_dir / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    logger.info("=" * 65)
    logger.info("TRAINING AND VALIDATION FINISHED")
    logger.info("=" * 65)
    logger.info("Best checkpoint: %s", best_checkpoint)
    logger.info("Metrics report: %s", summary_path)
    logger.info(
        "Measured validation metrics:\n%s",
        json.dumps(measured_metrics, indent=2),
    )
    logger.info(
        "Production approval: NOT GRANTED"
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        logger.warning(
            "Training interrupted by the user. "
            "Check the run directory for saved checkpoints."
        )
        sys.exit(130)

    except Exception:
        logger.exception(
            "SmartVision training failed. "
            "Review the error above before retrying."
        )
        sys.exit(1)

