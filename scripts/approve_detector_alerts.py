"""Record an authorized operator's explicit production-alert approval."""

import argparse
import getpass
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALERT_ENDPOINT_VERIFIED, YOLO_MODEL_PATH
from server.config import settings
from server.services import ntfy_service
from src.modeling.detectors.validation import (
    validation_report_sha256,
    validate_validation_record,
)


CONFIRMATION = "I reviewed the detector validation report and approve production alerts"


def approve_production_alerts(checkpoint_path, confirmation):
    if confirmation != CONFIRMATION:
        raise ValueError(
            "Approval requires the exact confirmation: "
            f"{CONFIRMATION!r}"
        )

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    approver = getpass.getuser().strip()
    if not approver or approver.casefold() not in settings.ALERT_APPROVERS:
        raise PermissionError(
            "The current Windows account is not authorized. Add its account "
            "name to SMARTVISION_ALERT_APPROVERS in the protected operator "
            "environment before approving."
        )

    transport_status, transport_error = ntfy_service.notification_state()
    if transport_status != "ready":
        raise ValueError(
            "Notification transport is not configured: "
            + (transport_error or transport_status)
        )
    if not ALERT_ENDPOINT_VERIFIED:
        raise ValueError(
            "The incident notification endpoint has not been explicitly "
            "verified. Set SMARTVISION_ALERT_ENDPOINT_VERIFIED=true only "
            "after confirming the backend integration."
        )

    status = validate_validation_record(checkpoint)
    if not status["passed"]:
        reasons = "\n".join(f"- {reason}" for reason in status["failures"])
        raise ValueError(
            "Production alerts cannot be approved until detector validation "
            f"passes:\n{reasons}"
        )
    if status["production_alerts_approved"]:
        raise ValueError("Production alerts are already explicitly approved.")

    try:
        ntfy_service.send_ntfy_notification(
            "SmartVision operator approval check. ntfy accepted a test "
            "notification request; phone receipt is not confirmed."
        )
    except ntfy_service.NtfyDeliveryError as exc:
        raise RuntimeError(
            f"ntfy test request failed; approval was not recorded: {exc}"
        ) from exc

    record = status["record"]
    record["production_alerts_approved"] = True
    record["production_alerts_approved_by"] = approver
    record["production_alerts_approved_at"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    record["production_alerts_approved_checkpoint_sha256"] = (
        record["checkpoint_sha256"]
    )
    record["production_alerts_validation_sha256"] = (
        validation_report_sha256(record)
    )
    record["production_alerts_test_provider_accepted"] = True
    record["production_alerts_test_provider_accepted_at"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    sidecar_path = Path(status["validation_record_path"])
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=sidecar_path.parent,
            delete=False,
        ) as temporary_file:
            json.dump(record, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, sidecar_path)
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return sidecar_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(YOLO_MODEL_PATH))
    parser.add_argument("--confirm", required=True)
    arguments = parser.parse_args()
    sidecar_path = approve_production_alerts(
        arguments.checkpoint,
        arguments.confirm,
    )
    print(f"Production alert approval recorded in: {sidecar_path}")
    print(
        "ntfy accepted a test request; this does not confirm phone receipt. "
        "This does not enable automatic delivery. Review the backend status, "
        "then separately configure SMARTVISION_AUTO_ALERTS_APPROVED=true "
        "and the notification/endpoint settings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
