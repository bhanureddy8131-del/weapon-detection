import os
import cv2
import numpy as np
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# --- PROJECT SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.config import ROOT_DIR, SEQUENCE_LENGTH
from src.modeling.classifiers.lstm import (
    POSE_FEATURE_LAYOUT,
    POSE_FEATURE_SIZE,
)
from src.modeling.pose_estimators.mediapipe import MediaPipeEstimator

# --- CONFIGURATION ---
VIDEOS_DIR = Path(os.getenv(
    "SMARTVISION_ACTION_VIDEO_DIR",
    str(Path.home() / "Desktop" / "dataset" / "split_clips"),
))
OUTPUT_DIR = Path(ROOT_DIR) / "data" / "processed" / "lstm_data"
DEBUG_VIDEO_DIR = Path(ROOT_DIR) / "data" / "processed" / "lstm_debug_videos"

STEP_SIZE = 10
YOLO_MODEL_SIZE = str(Path(ROOT_DIR) / "yolo11s.pt")

# THRESHOLDS
ENERGY_THRESHOLD = 0.08  # Min movement to be considered "Action" (Filters statues)
IOU_THRESHOLD = 0.3      # For simple tracking

def calculate_iou(box1, box2):
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)
    if x2_i < x1_i or y2_i < y1_i: return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0

def calculate_pose_energy(landmarks):
    """Simple heuristic: Sum of limb distances from body center."""
    if landmarks is None: return 0.0
    def get_pt(idx): return np.array([landmarks[idx*4], landmarks[idx*4+1]])
    
    wrists = [get_pt(15), get_pt(16)]
    shoulders = [get_pt(11), get_pt(12)]
    hips = [get_pt(23), get_pt(24)]
    
    torso_height = np.linalg.norm(shoulders[0] - hips[0]) + np.linalg.norm(shoulders[1] - hips[1])
    if torso_height == 0: return 0.0
    
    center = (hips[0] + hips[1]) / 2
    energy = 0
    for limb in wrists:
        energy += np.linalg.norm(limb - center) / torso_height
    return energy

def extract_features():
    if not VIDEOS_DIR.is_dir():
        raise FileNotFoundError(
            f"Labeled action clips were not found at {VIDEOS_DIR}. "
            "Set SMARTVISION_ACTION_VIDEO_DIR to the labeled clip directory."
        )

    class_directories = sorted(
        path for path in VIDEOS_DIR.iterdir() if path.is_dir()
    )
    class_names = {path.name.strip().lower() for path in class_directories}
    expected_classes = {"normal", "fighting", "shooting"}
    if class_names != expected_classes:
        raise ValueError(
            "Labeled clip folders must be exactly "
            f"{sorted(expected_classes)}; found {sorted(class_names)}."
        )

    print(f"⏳ Loading YOLO Model ({YOLO_MODEL_SIZE})...")
    yolo_model = YOLO(YOLO_MODEL_SIZE)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         yolo_model.to('mps')

    pose_estimator = MediaPipeEstimator()
    if not pose_estimator.available:
        raise RuntimeError(
            "MediaPipe Pose is unavailable; cannot create LSTM training data. "
            + pose_estimator.diagnostic
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_tasks = []
    classes = class_directories
    for class_directory in classes:
        class_name = class_directory.name.strip().lower()
        files = [
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov")
        ]
        for f in files:
            all_tasks.append({'path': str(f), 'class': class_name, 'name': f.name})
    if not all_tasks:
        raise ValueError(f"No labeled action video clips were found under {VIDEOS_DIR}.")
            
    for task in tqdm(all_tasks, desc="Multi-Person Extraction"):
        video_path, class_name, video_name = task['path'], task['class'], task['name']
        cap = cv2.VideoCapture(video_path)
        
        # Debug Video
        width, height = int(cap.get(3)), int(cap.get(4))
        debug_path = os.path.join(DEBUG_VIDEO_DIR, class_name, f"debug_{video_name}")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        out = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(5)), (width, height))
        
        # Missing pose frames split a track into separate valid sequences.
        active_tracks = {}
        next_track_id = 0
        
        # PREVIOUS BOXES: { track_id: box } for IoU matching
        prev_boxes = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = yolo_model(frame, classes=[0], verbose=False)
            
            current_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_boxes.append((x1, y1, x2, y2))
            
            # --- SIMPLE TRACKING (Match to previous frame) ---
            matched_ids = set()
            new_prev_boxes = {}
            
            for curr_box in current_boxes:
                # Find best match in prev_boxes
                best_iou = 0
                best_id = -1
                
                for tid, prev_box in prev_boxes.items():
                    if tid in matched_ids: continue # Already matched
                    iou = calculate_iou(curr_box, prev_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = tid
                
                # Assign ID
                if best_iou > IOU_THRESHOLD:
                    track_id = best_id
                    matched_ids.add(track_id)
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    active_tracks[track_id] = [] # Start new history
                
                new_prev_boxes[track_id] = curr_box
                
                # --- PROCESS PERSON ---
                x1, y1, x2, y2 = curr_box
                h, w, _ = frame.shape
                pad = 10
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
                crop = frame[cy1:cy2, cx1:cx2]
                
                landmarks = None
                energy = 0.0
                if crop.size > 0:
                    landmarks, _ = pose_estimator.predict(crop)
                    if (
                        landmarks is not None
                        and len(landmarks) == POSE_FEATURE_SIZE
                        and np.isfinite(landmarks).all()
                    ):
                        energy = calculate_pose_energy(landmarks)
                    else:
                        landmarks = None
                
                # SAVE DATA (Only if we have landmarks)
                active_tracks[track_id].append(
                    {"lm": landmarks, "energy": energy}
                    if landmarks is not None
                    else None
                )

                # DEBUG DRAW
                color = (0, 255, 0) if energy > ENERGY_THRESHOLD else (0, 100, 100) # Bright Green = Saved
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} E:{energy:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            prev_boxes = new_prev_boxes
            out.write(frame)

        cap.release()
        out.release()
        
        # --- SAVE COMPLETE WINDOWS OF CONSECUTIVE VALID POSES ---
        for tid, track_data in active_tracks.items():
            valid_segments = []
            current_segment = []
            for pose_frame in track_data:
                if pose_frame is None:
                    if current_segment:
                        valid_segments.append(current_segment)
                        current_segment = []
                    continue
                current_segment.append(pose_frame["lm"])
            if current_segment:
                valid_segments.append(current_segment)

            class_output_dir = OUTPUT_DIR / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            num_sequences = 0
            for segment in valid_segments:
                if len(segment) < SEQUENCE_LENGTH:
                    continue
                data_points = np.asarray(segment, dtype=np.float32)
                for start in range(
                    0,
                    len(data_points) - SEQUENCE_LENGTH + 1,
                    STEP_SIZE,
                ):
                    sequence = data_points[start:start + SEQUENCE_LENGTH]
                    if sequence.shape != (SEQUENCE_LENGTH, POSE_FEATURE_SIZE):
                        continue
                    save_name = (
                        f"{class_name}_{Path(video_name).stem}_"
                        f"id{tid}_{num_sequences}.npy"
                    )
                    np.save(class_output_dir / save_name, sequence)
                    num_sequences += 1

    pose_estimator.close()

if __name__ == "__main__":
    extract_features()