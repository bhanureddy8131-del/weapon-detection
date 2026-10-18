
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Ultralytics is not installed in this Python environment.")
    sys.exit(1)

search_roots = [
    ROOT / "models",
    ROOT / "runs" / "detect",
]

checkpoints = sorted({
    p.resolve()
    for folder in search_roots
    if folder.exists()
    for p in folder.rglob("*.pt")
    if p.is_file()
})

# Include the general-purpose model for comparison.
fallback = ROOT / "yolo11s.pt"
if fallback.is_file():
    checkpoints = sorted(set(checkpoints) | {fallback.resolve()})

if not checkpoints:
    print("No .pt checkpoint files found.")
    sys.exit(1)

print(f"\nFound {len(checkpoints)} checkpoint(s).\n")

for path in checkpoints:
    print("=" * 75)
    print(f"FILE: {path}")
    print(f"SIZE: {path.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        model = YOLO(str(path))
        names = model.names
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]

        print(f"CLASS COUNT: {len(names)}")
        print(f"CLASSES: {names}")

        normalized = {str(name).strip().lower() for name in names}
        required = {"atm", "backpack", "gun", "handbag", "knife", "person"}
        missing = required - normalized

        if not missing:
            print("RESULT: All six expected project classes are present.")
        else:
            print(f"RESULT: Missing expected project classes: {sorted(missing)}")

    except Exception as exc:
        print(f"LOAD ERROR: {type(exc).__name__}: {exc}")

print("\nInspection complete. No model files were modified.")
