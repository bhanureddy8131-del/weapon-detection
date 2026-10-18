@'
from pathlib import Path
import json
import torch
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "The_final-4" / "data.yaml"
BASE_MODEL = ROOT / "yolo11s.pt"
OUTPUT_DIR = ROOT / "runs" / "detect"
RUN_NAME = "smartvision_cpu_retrain"

EXPECTED_CLASSES = [
    "atm", "backpack", "gun", "handbag", "knife", "person"
]

def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")

    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Base YOLO model not found: {BASE_MODEL}")

    with DATA.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names", {})
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    else:
        names = list(names)

    if [str(x).lower() for x in names] != EXPECTED_CLASSES:
        raise ValueError(
            f"Dataset classes do not match expected classes.\n"
            f"Found: {names}\nExpected: {EXPECTED_CLASSES}"
        )

    dataset_root = Path(config.get("path", DATA.parent))
    if not dataset_root.is_absolute():
        dataset_root = (DATA.parent / dataset_root).resolve()

    for split in ("train", "val"):
        split_path = Path(config[split])
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.exists():
            raise FileNotFoundError(
                f"{split} dataset path does not exist: {split_path}"
            )

    print("Device: CPU")
    print(f"Dataset: {DATA}")
    print(f"Starting weights: {BASE_MODEL}")
    print("Classes:", ", ".join(EXPECTED_CLASSES))
    print("Training is starting. CPU training may take many hours.")

    model = YOLO(str(BASE_MODEL))

    model.train(
        data=str(DATA),
        epochs=20,
        imgsz=416,
        batch=4,
        workers=2,
        device="cpu",
        pretrained=True,
        optimizer="AdamW",
        patience=5,
        close_mosaic=5,
        cache=False,
        amp=False,
        plots=True,
        seed=42,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        save=True,
        save_period=5,
        verbose=True,
    )

    best_weights = OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Training finished without best.pt at {best_weights}"
        )

    print(f"\nTraining checkpoint: {best_weights}")
    print("\nRunning validation on the trained checkpoint...")

    trained_model = YOLO(str(best_weights))
    metrics = trained_model.val(
        data=str(DATA),
        imgsz=416,
        batch=4,
        device="cpu",
        workers=2,
        plots=True,
        project=str(OUTPUT_DIR),
        name=f"{RUN_NAME}_validation",
        exist_ok=True,
    )

    results = {
        str(key): float(value)
        for key, value in metrics.results_dict.items()
    }

    report_path = OUTPUT_DIR / RUN_NAME / "validation_metrics.json"
    report_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nVALIDATION METRICS")
    print(json.dumps(results, indent=2))
    print(f"\nMetrics saved to: {report_path}")
    print(f"Candidate weights: {best_weights}")
    print(
        "\nIMPORTANT: This is a candidate model, not an approved "
        "production detector. Keep threat alerts disabled until "
        "independent validation and alert-endpoint checks pass."
    )

if __name__ == "__main__":
    main()
'@ | Set-Content -Encoding UTF8 ".\scripts\train_smartvision_cpu.py"