from pathlib import Path
import yaml
from ultralytics import YOLO

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
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")

    with DATA.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names", {})
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    else:
        names = list(names)

    if [str(x).lower() for x in names] != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected dataset classes: {names}")

    print("Dataset:", DATA)
    print("Starting weights:", BASE_MODEL)
    print("Classes:", names)
    print("This script has passed its initial configuration checks.")
    print("No training has been started.")

if __name__ == "__main__":
    main()
