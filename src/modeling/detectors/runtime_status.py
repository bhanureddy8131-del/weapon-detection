"""Small local status handoff from the inference process to the API."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)
STATUS_PATH = Path(ROOT_DIR) / "logs" / "detector_runtime_status.json"


def publish_detector_status(detector):
    status = {
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "requested_checkpoint_path": (
            str(detector.requested_model_path)
            if detector.requested_model_path is not None
            else None
        ),
        "loaded_checkpoint_path": (
            str(detector.model_path) if detector.model_path is not None else None
        ),
        "loaded_checkpoint_sha256": detector.loaded_checkpoint_sha256,
        "custom_model_loaded": bool(
            detector.is_custom_model and detector.model is not None
        ),
        "model_validation_passed": bool(detector.is_validated),
        "production_alerts_approved": bool(
            detector.production_alerts_approved
        ),
        "diagnostic": detector.model_diagnostic,
    }
    temporary_path = STATUS_PATH.with_suffix(
        f".{os.getpid()}.tmp"
    )
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(json.dumps(status), encoding="utf-8")
        os.replace(temporary_path, STATUS_PATH)
    except OSError:
        logger.exception("Could not publish detector runtime status.")
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove detector runtime status temp file.")


def read_detector_status(expected_checkpoint_path, expected_checkpoint_sha256=None):
    expected_path = str(Path(expected_checkpoint_path).resolve())
    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        pid = status.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
            raise ValueError("Detector runtime status has an invalid process ID.")
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.argtypes = [
                ctypes.c_ulong,
                ctypes.c_bool,
                ctypes.c_ulong,
            ]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            process = kernel32.OpenProcess(0x1000, False, pid)
            if not process:
                if ctypes.get_last_error() != 5:
                    return {
                        "loaded": False,
                        "reason": "The inference process is not running.",
                    }
            else:
                kernel32.CloseHandle(process)
        else:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return {
                    "loaded": False,
                    "reason": "The inference process is not running.",
                }
            except PermissionError:
                pass

        if status.get("requested_checkpoint_path") != expected_path:
            return {
                "loaded": False,
                "reason": "The running inference process uses another checkpoint.",
            }
        if (
            expected_checkpoint_sha256 is not None
            and status.get("loaded_checkpoint_sha256")
            != expected_checkpoint_sha256
        ):
            return {
                "loaded": False,
                "status": status,
                "reason": (
                    "The running inference process loaded different checkpoint "
                    "contents from the validated checkpoint."
                ),
            }
        return {
            "loaded": status.get("custom_model_loaded") is True,
            "status": status,
            "reason": (
                None
                if status.get("custom_model_loaded") is True
                else status.get("diagnostic")
                or "The running inference process has no custom model loaded."
            ),
        }
    except FileNotFoundError:
        return {
            "loaded": False,
            "reason": "The inference application has not reported a loaded detector.",
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Detector runtime status is unavailable: %s", exc)
        return {
            "loaded": False,
            "reason": "The inference application runtime status is unavailable.",
        }
