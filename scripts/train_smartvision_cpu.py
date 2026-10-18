from pathlib import Path
import json
import yaml
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "The_final-4" / "data.yaml"
BASE_MODEL = ROOT / "yolo11s.pt"
RUN_NAME = "smartvision_cpu_retrain"
OUTPUT_DIR = ROOT / "runs" / "detect"

EXPECTED_CLASSES = ["atm", "backpack", "gun", "handbag", "knife", "person"]


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")

    with DATA.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names", {})
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    else:
        names = list(names)

    if [str(n).lower() for n in names] != EXPECTED_CLASSES:
        raise ValueError(f"Dataset classes do not match expected classes: {names}")

    dataset_root = Path(config.get("path", DATA.parent))
    if not dataset_root.is_absolute():
        dataset_root = (DATA.parent / dataset_root).resolve()

    for split in ("train", "val"):
        split_path = Path(config[split])
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"{split} directory not found: {split_path}")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Dataset:", DATA)
    print("Classes:", names)
    print("Starting CPU training. This may take many hours.")

    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATA),
        epochs=20,
        imgsz=416,
        batch=4,
        workers=2,
        device="cpu",
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
        raise FileNotFoundError(f"Training checkpoint not found: {best_weights}")

    print("\nValidating the trained checkpoint...")
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

    results = {str(k): float(v) for k, v in metrics.results_dict.items()}
    report = OUTPUT_DIR / RUN_NAME / "validation_metrics.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nVALIDATION RESULTS")
    print(json.dumps(results, indent=2))
    print("\nBest checkpoint:", best_weights)
    print("Metrics report:", report)
    print("Keep production alerts disabled until the detector is independently validated.")


if __name__ == "__main__":
    main()
