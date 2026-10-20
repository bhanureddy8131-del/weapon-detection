# Technical Architecture & Calculations

This document details the technical implementation, data flow, and calculations used in the SmartVision threat detection pipeline.

## System Overview

The pipeline processes video frames sequentially through five main stages:
1.  **Object Detection (YOLO)**
2.  **Object Tracking (ByteTrack)**
3.  **Pose Estimation (MediaPipe)**
4.  **Action Recognition (LSTM)**
5.  **Threat Logic (Rule-Based Matrix)**

---

## 1. Object Detection (YOLOv11)

*   **Model**: The configured custom six-class checkpoint is `models/yolo11s_fine_tune_v3.pt`. If it is unavailable, the loader uses the root `yolo11s.pt` general-purpose COCO checkpoint. The current workspace has the COCO fallback only; it cannot detect the project's `gun` or `atm` classes, and generic COCO `knife` is not treated as the project's validated knife class.
*   **Input**: Raw BGR Frame (Resolution Varies).
*   **Output**: Bounding boxes `[x1, y1, x2, y2]`, Confidence Score `conf`, Class ID `cls`.
*   **Configuration**:
    *   `CONFIDENCE_THRESHOLD`: **0.10** by default (`SMARTVISION_CONFIDENCE_THRESHOLD` can override it). Per-class thresholds may raise the effective minimum; the dashboard cannot lower it below the configured global minimum.
    *   `SMARTVISION_YOLO_MODEL_PATH`: optional trained checkpoint override. The missing configured checkpoint is reported; a generic fallback is not described as a weapon detector.
    *   `YOLO_IMAGE_SIZE`: **416** by default. The NMS IoU threshold defaults to **0.70**.
    *   **Custom checkpoint classes** (only when those weights are installed):
        *   `0`: ATM
        *   `1`: Backpack
        *   `2`: Gun
        *   `3`: Handbag
        *   `4`: Knife
        *   `5`: Person

### Calculations
*   **Non-Maximum Suppression (NMS)**: Performed internally by YOLO to remove duplicate boxes.
*   **Filtering**: Detections must meet the configured global and per-class confidence thresholds. Class names from the loaded weights are mapped explicitly; unrelated COCO classes are not promoted to project threat classes.

---

## 2. Object Tracking (ByteTrack)

*   **Library**: `trackers.ByteTrackTracker`, using `supervision.Detections` as its input/output contract.
*   **Input**: List of Person Detections `([x1, y1, x2, y2], conf, class_id=5)`.
*   **Output**: Unique `track_id` for each person across frames.

### Parameters & Calculations
*   **Track Activation Threshold**: **0.25**. Low-confidence detections can associate to existing tracks but do not start new tracks.
*   **Lost Track Buffer**: **60 frames at a 30 FPS reference rate**, equivalent to a two-second lost-track budget. Capture timestamps are passed to the tracker so skipped frames and irregular timing advance track state by elapsed time.
*   **Minimum IoU Similarity**: **0.8**. Controls how strictly detections must overlap predicted track locations.
*   **Track IDs**: A detection may be returned without a confirmed ID on its first observation; the ID is assigned after a consecutive match. Invalid and duplicate IDs are filtered before pose/action history is updated.
*   **Kalman Filter**: Used internally to predict the next position of a track based on velocity.

---

## 3. Pose Estimation (MediaPipe)

*   **Model**: `MediaPipe Pose` (Google).
*   **Input**: Cropped image of the **Person** (defined by Tracker Bounding Box).
*   **Output**: 33 Skeletal Landmarks.

### Calculations
*   **Data Layout**: Each landmark has `(x, y, z, visibility)`.
    *   `x, y`: Normalized coordinates [0.0, 1.0].
    *   `z`: Approximate depth.
    *   `visibility`: Confidence score [0.0, 1.0].
*   **Feature Vector**: Flattened `float32` array of size **132** (`33 landmarks * 4 values`).
    *   A missing or invalid pose clears that person's partial sequence. No action prediction is made until 30 consecutive valid pose frames have been collected.

---

## 4. Action Recognition (LSTM)

*   **Model**: Bidirectional LSTM (PyTorch).
*   **Input**: Sequence of **30 Frames** of Pose Feature Vectors.
    *   Tensor Shape: `(1, 30, 132)`
*   **Output**: Probability distribution over Action Classes.

### Architecture
*   **Input Size**: 132
*   **Hidden Size**: Defined in checkpoint (typically 64 or 128).
*   **Layers**: Defined in checkpoint (typically 2).
*   **Bidirectional**: Yes (Effective Hidden Size = `Hidden * 2`).
*   **Classifier**: Fully Connected Layer mapping to Action Classes.

### Calculations
*   **Sliding Window**: A `deque` (`maxlen=30`) maintains the history for each `track_id`. Newest frame is appended; oldest is dropped.
*   **Softmax**: Applied to model output logits to get probabilities.
*   **Confidence Threshold**: `ACTION_CONFIDENCE_THRESHOLD = 0.85` by default.
    *   An action is accepted when its maximum softmax probability meets the configured threshold.
    *   Below threshold, no action prediction is emitted; threat rules use `normal` for that frame. This is also the behavior while a valid sequence is still being collected.
*   **Checkpoint contract**: The loader requires a compatible bidirectional architecture, the configured sequence and feature layout, strict tensor shapes, and exactly the `normal`, `fighting`, and `shooting` class map. Action recognition remains unavailable when the configured checkpoint is missing or incompatible.

## 6. Incident Storage and Notification Delivery

The client derives an idempotency key from source, track ID, event type, class, and the configured cooldown window. It reuses the same key for every retry of a queued incident. FastAPI checks the optional key stored in `IncidentData` before writing an image or database row; a duplicate returns the existing incident and does not enqueue notifications again. The client worker always queues validated rule incidents for API storage even when external notifications are disabled. A success response must include a persisted incident ID; invalid or failed responses remain visible as failed in the session delivery view.

FastAPI stores incidents in the project-root `data/incidents.db` SQLite file by default. Both the API and dashboard use this absolute database path regardless of launch working directory; `SMARTVISION_DATABASE_PATH` can override it. Each incident stores its event time, detected class, confidence, tracking ID, detector notification authorization, and ntfy status/error. Existing databases are migrated on startup.

ntfy is opt-in (`SMARTVISION_ALERTS_ENABLED=false` by default) and additionally requires a topic, a currently loaded checkpoint whose fresh validation sidecar matches the checkpoint SHA-256 and held-out dataset, explicit production approval in that sidecar, `SMARTVISION_AUTO_ALERTS_APPROVED=true`, and the verified-endpoint gate. `scripts/validate_detector.py` performs the actual dataset evaluation separately from inference; `scripts/approve_detector_alerts.py` records a local operator's approval only after all checks pass. The API applies threat-rule and notification-policy checks before scheduling delivery, and rechecks them before sending. Blocked and successful attempts are persisted with the incident. ntfy uses a bounded request timeout; its topic is omitted from logs and errors. Only a successful 2xx response is recorded as `sent`; provider failures update the incident without deleting it. The safe notification test endpoint checks transport only and does not validate the model, approve production alerts, or create an incident.

## Dataset Audit, Training, and Validation

`scripts/audit_dataset.py` scans the configured train, validation, and held-out test splits without modifying source images or labels. It verifies the six-class map, image readability, label correspondence, normalized detection boxes and segmentation polygons, exact image duplicates, cross-split leakage, class counts, and annotation visualizations. Empty labels are reported as negative examples. Review the JSON report and suspected basename matches before training.

`scripts/train_weapon_improved.py` requires a passing source audit, creates a unique run directory, and builds a separate detection-format staging dataset. Valid polygon labels are converted into enclosing axis-aligned YOLO boxes in staging only; image files are hard-linked when supported and copied otherwise. The source dataset remains untouched. The script records software versions, seed, augmentation and training parameters, dataset identities, source checkpoint, and staging counts. Training output is a candidate only and is never selected or approved automatically. Resume reuses the exact staging dataset recorded for that run and refuses a changed source YAML.

`scripts/validate_detector.py` is a separate operation that loads the named checkpoint, binds the report to its exact path and SHA-256, dataset YAML and image/label manifest, class map, and evaluation settings, evaluates actual Ultralytics metrics, records per-class values and confusion data, times inference separately, and checks labeled representative images. The shared verifier requires explicit checks, validates the recorded settings and current dataset identity, and reconciles representative per-image counts against reported totals. `--split val` writes the checkpoint sidecar used for runtime gating. `--split test` writes a final held-out report without replacing that sidecar. Missing metrics, incomplete evaluation, mismatched class names, changed dataset identity, failed thresholds, or missing representative classes fail closed. Production approval is a distinct authorized operator operation and automatic alert dispatch remains a separate configuration gate.

The Streamlit dashboard reads saved validation and dataset-audit reports rather than deriving or inventing metrics. Its Video analysis page uses the same bounded capture/inference workers and reports processed-frame progress where the video source exposes a frame count. Incident history and export use the backend's persistent SQLite data.

---

## 5. Threat Logic Manager

*   **Input**:
    *   **Action Label** (from LSTM).
    *   **Visible Objects** (List of Class IDs from current frame).
    *   **Duration** (Time elapsed since `track_id` first appeared).
    *   **Unattended Bag Timer** (Time duration where `Bags > 0` AND `Persons == 0`).

### Logic Matrix
(See `docs/system_behavior.md` for the full matrix).

### Calculations
*   **Person Duration**: `current_time - person_start_time`.
*   **Unattended Bag Timer**: Increments by `dt` (frame time delta) whenever `has_bag` is True and `num_persons` is 0. Resets to 0 otherwise.
*   **Loitering Check**: `has_atm` AND `duration > 60.0s`.
*   **Unattended Check**: `unattended_bag_timer > 60.0s`.
