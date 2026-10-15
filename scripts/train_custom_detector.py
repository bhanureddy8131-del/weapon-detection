from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path(r"C:\weapon detection\wepondetection")
DATA = ROOT / "The_final-4" / "data.yaml"
BASE = ROOT / "yolo11s.pt"

if not DATA.exists():
    raise FileNotFoundError(f"Dataset config missing: {DATA}")
if not BASE.exists():
    raise FileNotFoundError(f"Base YOLO model missing: {BASE}")

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Training device: {device}")
print(f"Dataset config: {DATA}")

model = YOLO(str(BASE))
model.train(
    data=str(DATA),
    epochs=30,
    imgsz=416,
    batch=4,
    device=device,
    workers=2,
    patience=8,
    pretrained=True,
    project=str(ROOT / "SmartVision_Final"),
    name="weapon_detector",
    exist_ok=True,
    save=True,
    plots=True,
)

print("Training finished.")
print("Check:", ROOT / "SmartVision_Final" / "weapon_detector" / "weights" / "best.pt")
