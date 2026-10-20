import logging
import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    FastAPI,
    Depends,
    BackgroundTasks,
    File,
    UploadFile,
    Form,
    Request,
    HTTPException,
    Query,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from . import models, schemas, database
from .config import ROOT_DIR, settings
from .services import email_service, ntfy_service, telegram_service
from src.config import (
    ALERT_ENDPOINT_VERIFIED,
    ROOT_DIR as SOURCE_ROOT_DIR,
    VALIDATION_DATA_PATH,
    YOLO_MODEL_PATH,
)
from src.modeling.detectors.runtime_status import read_detector_status
from src.modeling.detectors.validation import validate_validation_record


# =========================================================
# LOGGING AND STORAGE
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartVisionServer")

ALERT_STORAGE_DIR = ROOT_DIR / "server" / "static" / "alerts"
ALERT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Explicit operator approval is still required for production
# automatic alerts. Never bypass detector validation.
AUTO_ALERTS_APPROVED = (
    os.getenv("SMARTVISION_AUTO_ALERTS_APPROVED", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

_VALIDATION_CACHE_LOCK = threading.Lock()
_VALIDATION_CACHE = {
    "checked_at": 0.0,
    "key": None,
    "status": None,
}

models.Base.metadata.create_all(bind=database.engine)
database.ensure_incident_columns()

app = FastAPI(title="SmartVision Backend", version="1.1.0")

app.mount(
    "/static",
    StaticFiles(directory=str(ROOT_DIR / "server" / "static")),
    name="static",
)


# =========================================================
# REQUEST VALIDATION
# =========================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    logger.error("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# =========================================================
# CONFIGURED MODEL
# =========================================================

def _configured_checkpoint_path():
    checkpoint = Path(YOLO_MODEL_PATH).expanduser()

    if not checkpoint.is_absolute():
        checkpoint = Path(SOURCE_ROOT_DIR) / checkpoint

    return checkpoint.resolve()


# =========================================================
# NOTIFICATION MESSAGE
# =========================================================

def _notification_message(
    incident: schemas.IncidentCreate,
    created_at: datetime,
) -> str:
    detected_class = incident.detected_class or incident.label
    confidence = f"{incident.confidence:.1%}"

    event_time = created_at

    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)

    event_time = event_time.isoformat(timespec="seconds")

    return (
        f"SmartVision security alert: {incident.threat_level}. "
        f"Detected class: {detected_class}. "
        f"Confidence: {confidence}. "
        f"Camera: {incident.camera_id}. "
        f"Time: {event_time}."
    )


# =========================================================
# DETECTOR VALIDATION
# =========================================================

def _detector_validation_status():
    checkpoint = _configured_checkpoint_path()

    sidecar = checkpoint.with_suffix(
        checkpoint.suffix + ".validation.json"
    )

    data_yaml = Path(VALIDATION_DATA_PATH).expanduser().resolve()

    def file_signature(path):
        try:
            metadata = path.stat()

            return (
                str(path),
                metadata.st_size,
                metadata.st_mtime_ns,
            )

        except OSError:
            return (str(path), None, None)

    cache_key = (
        file_signature(checkpoint),
        file_signature(sidecar),
        file_signature(data_yaml),
    )

    now = time.monotonic()

    with _VALIDATION_CACHE_LOCK:
        if (
            _VALIDATION_CACHE["key"] == cache_key
            and now - _VALIDATION_CACHE["checked_at"] < 10.0
        ):
            return _VALIDATION_CACHE["status"]

        try:
            status = validate_validation_record(
                checkpoint,
                data_yaml=data_yaml,
            )

        except (OSError, ValueError) as exc:
            logger.exception(
                "Could not verify detector validation record."
            )

            status = {
                "passed": False,
                "failures": [
                    f"Validation verification failed: {exc}"
                ],
                "production_alerts_approved": False,
                "production_alerts_approval_invalid": False,
                "record": {},
            }

        _VALIDATION_CACHE.update({
            "checked_at": now,
            "key": cache_key,
            "status": status,
        })

        return status


# =========================================================
# AUTOMATIC NOTIFICATION POLICY
# =========================================================

def _notification_policy():
    validation = _detector_validation_status()

    runtime = read_detector_status(
        _configured_checkpoint_path(),
        validation.get("record", {}).get("checkpoint_sha256"),
    )

    model_loaded = bool(runtime.get("loaded"))

    model_validated = bool(validation.get("passed"))

    running_model_validated = bool(
        runtime.get("status", {}).get("model_validation_passed")
    )

    production_approved = bool(
        validation.get("production_alerts_approved")
    )

    running_production_approved = bool(
        runtime.get("status", {}).get("production_alerts_approved")
    )

    transport_status, transport_error = (
        ntfy_service.notification_state()
    )

    configuration_enabled = bool(settings.ALERTS_ENABLED)

    if not model_validated:
        blocked_reason = (
            "Detector validation failed: "
            + "; ".join(validation.get("failures", []))
        )
        blocked_status = "blocked_validation"

    elif not model_loaded:
        blocked_reason = (
            runtime.get("reason")
            or (
                "The validated custom detector is not loaded "
                "by the inference process."
            )
        )
        blocked_status = "blocked_validation"

    elif not running_model_validated:
        blocked_reason = (
            "The inference process must restart to load the "
            "currently validated detector record."
        )
        blocked_status = "blocked_validation"

    elif not production_approved:
        if validation.get("production_alerts_approval_invalid"):
            blocked_reason = (
                "Production-alert approval metadata is missing "
                "or invalid. An authorized operator must review "
                "and approve the report."
            )
        else:
            blocked_reason = (
                "Production alerts have not been explicitly "
                "approved by an authorized operator."
            )

        blocked_status = "blocked_operator_approval"

    elif not running_production_approved:
        blocked_reason = (
            "The inference process must restart to load the "
            "operator's production-alert approval."
        )
        blocked_status = "blocked_operator_approval"

    elif not AUTO_ALERTS_APPROVED:
        blocked_reason = (
            "Automatic alerts are not enabled. Set "
            "SMARTVISION_AUTO_ALERTS_APPROVED=true only after "
            "the separate production approval."
        )
        blocked_status = "blocked_operator_approval"

    elif not configuration_enabled:
        blocked_reason = (
            "Phone notification configuration is disabled. "
            "Set SMARTVISION_ALERTS_ENABLED=true to enable "
            "ntfy transport."
        )
        blocked_status = "blocked_configuration"

    elif transport_status != "ready":
        blocked_reason = (
            transport_error
            or "The ntfy notification transport is not ready."
        )
        blocked_status = "blocked_configuration"

    elif not ALERT_ENDPOINT_VERIFIED:
        blocked_reason = (
            "The FastAPI alert endpoint has not been "
            "explicitly verified."
        )
        blocked_status = "blocked_configuration"

    else:
        blocked_reason = None
        blocked_status = None

    return {
        "transport_status": transport_status,
        "transport_error": transport_error,
        "notification_configuration_enabled": (
            configuration_enabled
        ),
        "custom_model_loaded": model_loaded,
        "model_validation_passed": model_validated,
        "running_model_validation_passed": (
            running_model_validated
        ),
        "production_alerts_approved": production_approved,
        "running_production_alerts_approved": (
            running_production_approved
        ),
        "automatic_alerts_approved": AUTO_ALERTS_APPROVED,
        "automatic_alerts_enabled": blocked_reason is None,
        "blocked_reason": blocked_reason,
        "blocked_status": blocked_status,
        "validation_failures": validation.get("failures", []),
        "model_runtime_reason": runtime.get("reason"),
    }


# =========================================================
# INCIDENT NOTIFICATION AUTHORIZATION
# =========================================================

def _incident_notification_block(
    policy,
    requested_authorization=None,
):
    """
    Determine whether an incident can be sent automatically.

    The client-side authorization flag is not required.
    The server's complete production policy remains mandatory.
    """

    if policy["blocked_status"] is not None:
        return (
            policy["blocked_status"],
            policy["blocked_reason"],
        )

    return "pending", None


# =========================================================
# THREAT PAYLOAD VALIDATION
# =========================================================

def _validate_threat_incident(
    incident: schemas.IncidentCreate,
) -> str | None:
    """
    Check that the incident payload contains evidence required
    by its threat rule.

    Payload consistency does not prove detector accuracy.
    """

    level = incident.threat_level.strip().upper()

    action_data = incident.data.action_classification

    action = (
        action_data.class_.strip().lower()
        if action_data and action_data.class_
        else ""
    )

    detections = incident.data.detections or []

    detected_classes = {
        detection.class_.strip().lower()
        for detection in detections
        if detection.class_
    }

    violent_action = action in {"shooting", "violence"}

    weapon_detected = bool(
        detected_classes.intersection({"gun", "knife"})
    )

    spatial_context = incident.data.spatial_context

    atm_present = (
        spatial_context.atm_present
        if spatial_context is not None
        else False
    )

    atm_detected = atm_present and "atm" in detected_classes

    if level == "SAFE":
        return None

    if level == "HIGH: WEAPON DETECTED" and weapon_detected:
        return None

    if (
        level == "CRITICAL: SHOOTER"
        and "gun" in detected_classes
        and violent_action
    ):
        return None

    if (
        level == "CRITICAL: KNIFE ATTACK"
        and "knife" in detected_classes
        and violent_action
    ):
        return None

    if (
        level == "CRITICAL: ATM ROBBERY"
        and atm_detected
        and violent_action
    ):
        return None

    if level == "WARNING: FIGHTING" and action == "violence":
        return None

    if (
        level == "WARNING: SUSPICIOUS STANCE"
        and action == "shooting"
    ):
        return None

    if level == "WARNING: LOITERING AT ATM" and atm_detected:
        return None

    if (
        level == "WARNING: UNATTENDED BAGGAGE"
        and action == "unattended"
        and detected_classes.intersection({"backpack", "handbag"})
    ):
        return None

    return (
        "Incident does not match a supported validated threat rule."
    )


# =========================================================
# SECONDARY TELEGRAM AND EMAIL NOTIFICATIONS
# =========================================================

def process_notifications(
    incident: schemas.IncidentCreate,
    image_path: str | None = None,
) -> None:
    """
    Send secondary Telegram/email notifications for incidents
    already authorized by the server's notification policy.
    """

    msg = (
        "SMARTVISION SECURITY ALERT\n"
        f"Threat: {incident.threat_level}\n"
        f"Detected class: "
        f"{incident.detected_class or incident.label}\n"
        f"Confidence: {incident.confidence:.1%}\n"
        f"Camera: {incident.camera_id}\n"
        f"Time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
    )

    for detection in incident.data.detections or []:
        class_name = detection.class_.strip().lower()

        if class_name in {"gun", "knife"}:
            msg += (
                f"Weapon detected: {class_name.upper()} "
                f"({detection.confidence:.1%})\n"
            )

    try:
        telegram_service.send_telegram_alert(msg, image_path)
    except Exception:
        logger.exception("Telegram notification failed.")

    if incident.threat_level.strip().upper().startswith(
        ("CRITICAL", "HIGH")
    ):
        try:
            email_service.send_email_alert(
                subject=(
                    f"SmartVision Threat Alert: "
                    f"{incident.threat_level}"
                ),
                body=msg,
                image_path=image_path,
            )
        except Exception:
            logger.exception("Email notification failed.")


# =========================================================
# PERSIST AND DELIVER PHONE NOTIFICATION
# =========================================================

def deliver_incident_notification(
    incident_id: str,
    message: str,
    session_factory=None,
) -> None:
    """
    Deliver a persisted incident notification only when the
    current server policy authorizes production delivery.
    """

    db = (session_factory or database.SessionLocal)()

    try:
        incident = db.get(models.Incident, incident_id)

        if incident is None:
            logger.warning(
                "Notification skipped: incident %s does not exist.",
                incident_id,
            )
            return

        if incident.threat_level.strip().upper() == "SAFE":
            incident.notification_status = "disabled"
            incident.notification_error = (
                "Safe events do not generate notifications."
            )
            db.commit()
            return

        # Recheck the complete policy at delivery time.
        policy = _notification_policy()

        delivery_status, delivery_error = (
            _incident_notification_block(
                policy,
                incident.notification_authorized,
            )
        )

        if delivery_status != "pending":
            incident.notification_status = delivery_status
            incident.notification_error = delivery_error
            db.commit()
            return

        incident.notification_status = "pending"
        incident.notification_error = None
        db.commit()

        try:
            ntfy_service.send_ntfy_notification(message)

        except ntfy_service.NtfyDeliveryError as exc:
            incident.notification_status = "failed"
            incident.notification_error = str(exc)
            db.commit()

            logger.error(
                "Phone notification failed for incident %s: %s",
                incident_id,
                exc,
            )
            return

        incident.notification_status = "sent"
        incident.notification_error = None
        db.commit()

        logger.info(
            "Phone notification sent for incident %s.",
            incident_id,
        )

    except SQLAlchemyError:
        db.rollback()
        logger.exception(
            "Could not persist notification state for incident %s.",
            incident_id,
        )

    except Exception:
        db.rollback()
        logger.exception(
            "Unexpected notification error for incident %s.",
            incident_id,
        )

    finally:
        db.close()


# =========================================================
# CREATE INCIDENT
# =========================================================

@app.post(
    "/api/v1/incidents",
    response_model=schemas.IncidentResponse,
)
def create_incident(
    background_tasks: BackgroundTasks,
    incident_data: str = Form(...),
    file: UploadFile = File(None),
    db: Session = Depends(database.get_db),
):
    try:
        payload = json.loads(incident_data)
        incident = schemas.IncidentCreate(**payload)

    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Invalid incident payload: %s", exc)

        raise HTTPException(
            status_code=422,
            detail="Incident payload is invalid.",
        ) from exc

    validation_error = _validate_threat_incident(incident)

    if validation_error:
        raise HTTPException(
            status_code=422,
            detail=validation_error,
        )

    threat_level = incident.threat_level.strip().upper()
    idempotency_key = incident.data.idempotency_key

    # Return an existing incident for duplicate requests.
    if idempotency_key:
        try:
            existing_incident = (
                db.query(models.Incident)
                .filter(
                    models.Incident.idempotency_key == idempotency_key
                )
                .first()
            )

        except SQLAlchemyError as exc:
            logger.exception("Could not check incident idempotency.")

            raise HTTPException(
                status_code=500,
                detail="Could not check for a duplicate incident.",
            ) from exc

        if existing_incident is not None:
            logger.info("Duplicate incident request ignored.")
            return existing_incident

    # Authorization comes from the server policy.
    # A client cannot bypass detector validation or approval.
    notification_policy = _notification_policy()

    incident_is_authorized = (
        notification_policy["automatic_alerts_enabled"]
        and threat_level != "SAFE"
    )

    logger.info(
        "Received incident: %s (%s)",
        incident.label,
        incident.threat_level,
    )

    # Save optional evidence snapshot.
    file_path = None

    if file is not None:
        try:
            extension = (
                os.path.splitext(file.filename or "")[1]
                .lstrip(".")
                or "jpg"
            )

            if not extension.isalnum():
                raise ValueError("Invalid snapshot file extension.")

            filename = f"{uuid.uuid4()}.{extension}"
            file_path = str(ALERT_STORAGE_DIR / filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        except (OSError, ValueError) as exc:
            logger.exception("Could not save incident snapshot.")

            raise HTTPException(
                status_code=500,
                detail="Could not save incident snapshot.",
            ) from exc

        finally:
            try:
                file.file.close()
            except Exception:
                pass

    created_at = models.utc_now()
    detected_class = incident.detected_class or incident.label

    action_classification = incident.data.action_classification

    tracking_id = (
        action_classification.track_id
        if action_classification is not None
        else None
    )

    if threat_level == "SAFE":
        notification_status = "disabled"
        notification_error = (
            "Safe events do not generate notifications."
        )

    else:
        notification_status, notification_error = (
            _incident_notification_block(notification_policy)
        )

    db_incident = models.Incident(
        created_at=created_at,
        camera_id=incident.camera_id,
        threat_level=incident.threat_level,
        label=incident.label,
        confidence=incident.confidence,
        idempotency_key=idempotency_key,
        detected_class=detected_class,
        tracking_id=(
            str(tracking_id) if tracking_id is not None else None
        ),
        notification_authorized=incident_is_authorized,
        notification_status=notification_status,
        notification_error=notification_error,
        data=incident.data.model_dump(by_alias=True),
        image_path=file_path,
    )

    db.add(db_incident)

    try:
        db.commit()

    except IntegrityError as exc:
        db.rollback()
        existing_incident = None

        if idempotency_key:
            try:
                existing_incident = (
                    db.query(models.Incident)
                    .filter(
                        models.Incident.idempotency_key == idempotency_key
                    )
                    .first()
                )

            except SQLAlchemyError as query_exc:
                logger.exception(
                    "Could not resolve incident idempotency conflict."
                )

                raise HTTPException(
                    status_code=500,
                    detail="Could not resolve incident write conflict.",
                ) from query_exc

        if existing_incident is not None:
            if file_path:
                try:
                    os.remove(file_path)
                except OSError:
                    logger.warning(
                        "Could not remove duplicate incident snapshot."
                    )

            return existing_incident

        logger.exception("Incident violated a database constraint.")

        if file_path:
            try:
                os.remove(file_path)
            except OSError:
                logger.warning("Could not remove unsaved snapshot.")

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save incident because of a database constraint."
            ),
        ) from exc

    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Could not persist incident.")

        if file_path:
            try:
                os.remove(file_path)
            except OSError:
                logger.warning("Could not remove unsaved snapshot.")

        raise HTTPException(
            status_code=500,
            detail="Could not save incident to persistent storage.",
        ) from exc

    try:
        db.refresh(db_incident)

    except SQLAlchemyError as exc:
        logger.exception(
            "Incident was saved but could not be read back."
        )

        raise HTTPException(
            status_code=500,
            detail="Incident was saved but could not be read back.",
        ) from exc

    # Schedule the phone notification only after the incident
    # has been persisted and the complete policy allows delivery.
    if (
        threat_level != "SAFE"
        and incident_is_authorized
        and notification_status == "pending"
    ):
        background_tasks.add_task(
            deliver_incident_notification,
            db_incident.id,
            _notification_message(incident, created_at),
            sessionmaker(bind=db.get_bind()),
        )

        # Secondary services use the same server authorization decision.
        background_tasks.add_task(
            process_notifications,
            incident,
            file_path,
        )

    return db_incident


# =========================================================
# INCIDENT HISTORY
# =========================================================

@app.get(
    "/api/v1/incidents",
    response_model=list[schemas.IncidentResponse],
)
def get_incidents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(database.get_db),
):
    try:
        return (
            db.query(models.Incident)
            .order_by(models.Incident.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    except SQLAlchemyError as exc:
        logger.exception("Could not read incident history.")

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not read incident history from persistent storage."
            ),
        ) from exc


@app.get("/api/v1/incidents/summary")
def get_incident_summary(
    db: Session = Depends(database.get_db),
):
    today = models.utc_now().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    try:
        total = (
            db.query(func.count(models.Incident.id)).scalar() or 0
        )

        today_total = (
            db.query(func.count(models.Incident.id))
            .filter(models.Incident.created_at >= today)
            .scalar()
            or 0
        )

        class_rows = (
            db.query(
                func.coalesce(
                    models.Incident.detected_class,
                    models.Incident.label,
                ).label("incident_class"),
                func.count(models.Incident.id).label("incident_count"),
            )
            .group_by("incident_class")
            .order_by("incident_class")
            .all()
        )

        return {
            "total": total,
            "today": today_total,
            "by_class": {
                incident_class: incident_count
                for incident_class, incident_count in class_rows
            },
        }

    except SQLAlchemyError as exc:
        logger.exception("Could not summarize incident history.")

        raise HTTPException(
            status_code=500,
            detail="Could not read incident summary from persistent storage.",
        ) from exc


# =========================================================
# NOTIFICATION STATUS AND TEST
# =========================================================

@app.get("/api/v1/notifications/status")
def get_notification_status():
    policy = _notification_policy()

    return {
        "status": policy["transport_status"],
        "transport_status": policy["transport_status"],
        "transport_error": policy["transport_error"],
        "error": policy["transport_error"],
        "enabled": policy["notification_configuration_enabled"],
        "notification_configuration_enabled": (
            policy["notification_configuration_enabled"]
        ),
        "custom_model_loaded": policy["custom_model_loaded"],
        "model_validation_passed": policy["model_validation_passed"],
        "running_model_validation_passed": (
            policy["running_model_validation_passed"]
        ),
        "production_alerts_approved": (
            policy["production_alerts_approved"]
        ),
        "running_production_alerts_approved": (
            policy["running_production_alerts_approved"]
        ),
        "automatic_alerts_approved": (
            policy["automatic_alerts_approved"]
        ),
        "automatic_alerts_enabled": (
            policy["automatic_alerts_enabled"]
        ),
        "blocked_reason": policy["blocked_reason"],
        "validation_failures": policy["validation_failures"],
    }


@app.post("/api/v1/notifications/test")
def test_notification():
    service_state, service_error = (
        ntfy_service.notification_state()
    )

    if service_state != "ready":
        raise HTTPException(
            status_code=409,
            detail=service_error or "Notification service is not ready.",
        )

    now = models.utc_now()

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    now = now.isoformat(timespec="seconds")

    try:
        ntfy_service.send_ntfy_notification(
            "SmartVision test notification. "
            f"Test delivery requested at {now}. "
            "This is a delivery test, not a confirmed threat."
        )

    except ntfy_service.NtfyDeliveryError as exc:
        logger.error("ntfy test notification failed: %s", exc)

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return {
        "status": "sent",
        "message": "ntfy accepted the test notification.",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/v1/health")
@app.get("/")
def health_check():
    database_error = None

    try:
        with database.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        database_status = "connected"

    except SQLAlchemyError:
        logger.exception("Database health check failed.")

        database_status = "unavailable"
        database_error = "Incident database is unavailable."

    notification_policy = _notification_policy()

    return {
        "status": "running",
        "database_status": database_status,
        "database_error": database_error,
        "notification_status": (
            notification_policy["transport_status"]
        ),
        "notification_error": (
            notification_policy["transport_error"]
        ),
        "notification_configuration_enabled": (
            notification_policy["notification_configuration_enabled"]
        ),
        "custom_model_loaded": (
            notification_policy["custom_model_loaded"]
        ),
        "model_validation_passed": (
            notification_policy["model_validation_passed"]
        ),
        "running_model_validation_passed": (
            notification_policy["running_model_validation_passed"]
        ),
        "production_alerts_approved": (
            notification_policy["production_alerts_approved"]
        ),
        "running_production_alerts_approved": (
            notification_policy["running_production_alerts_approved"]
        ),
        "automatic_alerts_approved": (
            notification_policy["automatic_alerts_approved"]
        ),
        "automatic_alerts_enabled": (
            notification_policy["automatic_alerts_enabled"]
        ),
        "notification_blocked_reason": (
            notification_policy["blocked_reason"]
        ),
    }