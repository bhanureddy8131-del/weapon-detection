# SmartVision Pro Integration Review

Review date: 2026-10-16

## Completed fixes

- Inference worker now stops its capture source if inference exits unexpectedly. A regression test covers a capture `read()` exception.
- Alert delivery shutdown no longer depends on inserting a sentinel into a potentially full bounded queue. Shutdown signals the worker, which drains queued incidents and exits; a blocked-send/full-queue test covers the path.
- Webcam reconnection test now waits for the first frame from the reopened source instead of assuming an open device has already published a frame.
- Migrated from deprecated `supervision.ByteTrack` to `trackers.ByteTrackTracker` 2.6.1. Timestamps now advance ByteTrack using monotonic capture time, low-confidence track association remains enabled, and IDs appear only after the tracker's observed confirmation step.
- Webcam capture now requests 1280x720; inference still downsizes to 640 pixels wide for CPU speed. Detection boxes are scaled onto the original-resolution annotated output. Processing timing includes overlay rendering.
- Live detection now shows source/inference resolution, actual detection/person/trained-weapon counts, FPS/latency, and a bounded recent-detection table. MediaPipe is not initialized when no LSTM is loaded.
- Corrupt custom YOLO checkpoints now fall back to generic COCO weights with an explicit diagnostic. Training scripts refuse to overwrite existing detector, metadata, or LSTM artifacts.
- Added `scripts/diagnose.py` for versions, compute device, actual checkpoints/classes, dataset class mapping, sample-image detections, and timed inference.
- FastAPI idempotency is now persistent and concurrency-safe: a unique nullable key is stored in SQLite, existing databases receive a startup column/index migration, duplicate requests return the original incident, and a uniqueness race does not schedule a second notification.
- Fixed the current inference-file regression that referenced undefined `x1/y1/x2/y2` while publishing tracked people; coordinates now come from the matching ByteTrack output row.
- Lowered ByteTrack's high/low confidence split to the existing 0.25 activation threshold. A valid person scored 0.378 by COCO was above the app's 0.10 detector threshold but previously could not start a track with a 0.50 split.
- Windows `start.bat` now binds FastAPI to `127.0.0.1`, consistent with `start.ps1` and the configured local API URL.
- README, environment setup, architecture, and this repair report document the current fallback behavior, diagnostics, platform requirements, source test checklist, tracker API, and alert idempotency.

## Verification results

- Python syntax: `compileall` passed for `src`, `pipeline`, `server`, `scripts`, and `tests`.
- Imports: `pipeline.app`, `pipeline.run_inference`, `server.main`, and `src.config` imported successfully.
- Dependencies: `pip check` reported no broken requirements after installing and declaring `trackers>=2.6.1,<3`. Tested with Python 3.14.4, Torch 2.14.1+cpu, Ultralytics 8.4.175, Supervision 0.30.9, Trackers 2.6.1, MediaPipe 1.1.0, OpenCV 5.0.0, FastAPI 0.143.0, and Streamlit 1.65.0.
- Existing and new tests: `python -m unittest discover -s tests -q` passed, 50 tests after the no-detection fixes.
- Syntax/imports: `compileall` passed for project Python modules; pipeline/dashboard/server/config imports passed; Pylance found no errors in touched runtime files after fixes.
- FastAPI: TestClient verified multipart incident creation, duplicate POST returning the same ID with one stored row, and the SQLite migration/index on an old table. Client-side tests simulated a successful response, timeout, connection failure, transient retries, and duplicate-event behavior. No production alert or external Telegram/email delivery was attempted.
- Dashboard: Streamlit AppTest rendered Overview and Model and system health without exceptions. A Live detection smoke test injected a worker snapshot and confirmed matching metrics, full-size image rendering, and two recent-detection rows. At review time the health page reported generic COCO, missing LSTM, disabled alerts, and the actual API connection state.
- Local YOLO/model mapping: only the root `yolo11s.pt` fallback is available. It was loaded and run on real local dataset images. COCO person/backpack map to project canonical IDs; generic COCO knife maps to `-1`. The six names in `The_final-4/data.yaml` match the expected training contract, but the loaded model is not that detector.
- No-detection regression: the same local image produced a person at 0.378 confidence. Before the tracker split adjustment it was discarded as low-confidence; after adjustment, two consecutive process_frame calls reported one tracked person with bbox `[241, 78, 292, 214]` and track ID `0`. This verifies that sample's person path, not weapon accuracy.
- Local pose: MediaPipe initialized from `models/pose_landmarker_lite.task`. A real person crop from a local image produced the expected 132 features in 63.97 ms; another smaller crop correctly produced no pose.
- Video input: a temporary 12-frame 1280x720 MJPG video was fully inferred (12/12 frames), with 640x360 inference and 1280x720 annotated output; the worker completed without error and capture thread stopped. This was generated from a local sample image, not a representative accuracy video.
- Webcam: device 0 ran at 1280x720 capture and 640x360 inference. Over a six-second run capture was 30.15 FPS; live-frame age mean 19.64 ms / p95 38.91 ms / max 73.96 ms; inference 10.78 FPS; latest capture-to-inference latency 110.45 ms and processing 78.47 ms. Twelve frames completed without worker error; missing LSTM meant MediaPipe was not initialized; both threads stopped. Recovery was unit-tested with a simulated disconnect, not a forced physical disconnect.
- Pose: MediaPipe initialized and produced a real 132-feature vector from a person crop in 63.97 ms. A smaller person crop returned no pose, which is a valid no-detection outcome.
- Same-image detector timing: the same local 640x640 test image and 416 input measured 107.93 ms mean / 117.21 ms p95 before and 103.51 ms mean / 107.46 ms p95 after, with two warmups and 12 timed runs. Both used generic YOLO weights; this small runtime difference is not an accuracy or meaningful speedup claim. The diagnostic's separate three-run smoke reported 117.93 ms mean / 132.02 ms p95 on another real test image.
- Git review: the working tree was clean before these review fixes. No tracked deletions were found in the recent integration changes, and a scan of recent committed text changes found no credential-like assignments. The recent commits include backup files, generated training plots, an empty `package-lock.json`, and a suspiciously named added file `mmmssm,`; these were left untouched because their removal was not requested. This review modified the pipeline, tracker/detector, server schema/storage/handler, training utilities, requirements, tests, startup binding, README, environment/architecture docs, and diagnostic/repair documentation; no checkpoint or Git history was changed.

## Remaining blockers and limits

- `models/yolo11s_fine_tune_v3.pt` is absent. The installed fallback is the general COCO model and does not support the project's trained `gun` or `atm` classes. This review does not establish custom weapon detection or custom-model accuracy.
- `models/lstm_action_recognition_pro_v2.pth` and `data/processed/lstm_data` are absent. LSTM sequence/checkpoint behavior is unit-tested, but fighting/shooting recognition is unavailable and no trained LSTM was tested.
- FastAPI incident receipt was tested with TestClient. Alert connection failures and retries were simulated with a mocked HTTP session. Telegram/email delivery and an actual production alert POST were not verified; production alerts remain disabled by the configured model/operator/endpoint gates.
- Older clients that omit `idempotency_key` remain accepted but do not receive server-side retry deduplication. New pipeline events include the key.
- The UI was checked headlessly with AppTest, not by browser screenshot or sustained multi-session use. Camera and inference timings are from one local five-second run and vary with scene, hardware, and model.
- The TestClient run emitted upstream Starlette/httpx, Pydantic, and SQLAlchemy deprecation/resource warnings; tests passed. Upgrade compatibility should be revisited before those dependencies remove the deprecated APIs.

## Files Modified

- `README.md`
- `docs/REPAIR_PLAN.md`
- `docs/env_setup.md`
- `docs/technical_architecture.md`
- `environment/requirements-base.txt`
- `pipeline/app.py`
- `pipeline/inference_worker.py`
- `pipeline/latest_frame_capture.py`
- `pipeline/run_inference.py`
- `scripts/diagnose.py`
- `scripts/train_lstm.py`
- `scripts/train_lstm_pro.py`
- `scripts/train_weapon_cpu.py`
- `server/database.py`
- `server/main.py`
- `server/models.py`
- `server/schemas.py`
- `src/core/threat_manager.py`
- `src/modeling/detectors/yolo_detector.py`
- `src/modeling/trackers/byte_tracker.py`
- `start.bat`
- `tests/test_incident_api.py`
- `tests/test_live_pipeline.py`
- `tests/test_lstm_action_recognition.py`
- `tests/test_pose_tracking.py`

## Exact Windows startup commands

From `C:\weapon detection\wepondetection`, open two PowerShell terminals. Use the existing `.venv`:

Terminal 1, FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Terminal 2, Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run pipeline\app.py --server.port 8501
```

The equivalent launcher is:

```powershell
.\start.ps1
```

The dashboard URL is `http://localhost:8501`; the local API is `http://127.0.0.1:8000`.

Run the local runtime/model diagnostic from the project root with:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose.py --runs 10
```
