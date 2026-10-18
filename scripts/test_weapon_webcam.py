
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "smartvision_diagnostic"
    / "cpu_3epoch"
    / "weights"
    / "best.pt"
)

CAMERA_ID = 0
IMAGE_SIZE = 320
CONFIDENCE = 0.10
WARMUP_RUNS = 2
MEASURED_RUNS = 8


def predict(model, frame):
    with torch.inference_mode():
        result = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
            device="cpu",
            max_det=50,
        )[0]
    return result


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    print(f"Loading model: {MODEL_PATH}")
    print("PyTorch:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())

    model = YOLO(str(MODEL_PATH))
    print("Classes:", model.names)

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    timings = []
    frame_count = 0

    try:
        # Get a frame before measuring.
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot read a webcam frame.")

        print("\nWarming up model...")
        for i in range(WARMUP_RUNS):
            t0 = time.perf_counter()
            result = predict(model, frame)
            elapsed = (time.perf_counter() - t0) * 1000
            print(
                f"Warm-up {i + 1}: {elapsed:.0f} ms; "
                f"detections={len(result.boxes)}"
            )

        print("\nMeasuring steady-state inference...")
        for i in range(MEASURED_RUNS):
            ok, frame = cap.read()
            if not ok:
                print("Could not read next webcam frame.")
                break

            t0 = time.perf_counter()
            result = predict(model, frame)
            elapsed = (time.perf_counter() - t0) * 1000
            timings.append(elapsed)
            frame_count += 1

            print(
                f"Run {i + 1}: {elapsed:.0f} ms | "
                f"detections={len(result.boxes)}"
            )

            # Display only the final measured frame.
            if i == MEASURED_RUNS - 1:
                cv2.imshow("SmartVision - Benchmark", result.plot())
                cv2.waitKey(1000)

        if timings:
            avg = sum(timings) / len(timings)
            print("\n--- BENCHMARK SUMMARY ---")
            print(f"Frames measured: {frame_count}")
            print(f"Average inference: {avg:.0f} ms")
            print(f"Best inference: {min(timings):.0f} ms")
            print(f"Worst inference: {max(timings):.0f} ms")
            print(f"Inference-only FPS: {1000 / avg:.2f}")

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
