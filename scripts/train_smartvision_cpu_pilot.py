from pathlib import Path
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "The_final-4" / "data.yaml"
BASE_MODEL = ROOT / "yolo11s.pt"
OUTPUT_DIR = ROOT / "runs" / "detect"
RUN_NAME = "smartvision_cpu_pilot_3epoch"

EXPECTED_CLASSES = [
    "atm", "backpack", "gun", "handbag", "knife", "person"
]


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")

    with DATA.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]

    if [str(name).lower() for name in names] != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected dataset classes: {names}")

    print("Starting YOLO11s pilot training")
    print("Dataset:", DATA)
    print("Output:", OUTPUT_DIR / RUN_NAME)
    print("Device: CPU")
    print("Epochs: 3")
    print("Batch size: 2")
    print("Existing checkpoints will not be overwritten.")

    model = YOLO(str(BASE_MODEL))

    model.train(
        data=str(DATA),
        epochs=3,
        imgsz=416,
        batch=2,
        workers=0,
        device="cpu",
        pretrained=True,
        optimizer="AdamW",
        patience=3,
        close_mosaic=1,
        cache=False,
        amp=False,
        plots=False,
        seed=42,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=False,
        save=True,
        save_period=1,
        verbose=True,
    )

    best = OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    print("\nPilot training finished.")
    print("Best checkpoint:", best)
    print("Checkpoint exists:", best.is_file())
    print("This model is not approved for production alerts.")


if __name__ == "__main__":
    main()
