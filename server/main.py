import logging
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
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


# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartVisionServer")

ALERT_STORAGE_DIR = ROOT_DIR / "server" / "static" / "alerts"
ALERT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the persistent store using the same root-relative database as the dashboard.
models.Base.metadata.create_all(bind=database.engine)
database.ensure_incident_columns()

app = FastAPI(title="SmartVision Backend", version="1.0.0")

# Serve static files for alert snapshots
app.mount(
    "/static",
    StaticFiles(directory=str(ROOT_DIR / "server" / "static")),
    name="static",
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()}")
    # logger.error(f"Body: {exc.body}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


def process_notifications(incident: schemas.IncidentCreate, image_path: str = None):
    msg = f" **SMARTVISION SECURITY ALERT** \n"
    msg += f"\n**Threat**: {incident.threat_level}\n"

    weapon_found = False
    if (
        "WEAPON" in incident.threat_level
        or "SHOOTER" in incident.threat_level
        or "KNIFE" in incident.threat_level
    ):
        detections = incident.data.detections or []
        for d in detections:
            if d.class_.lower() in ["gun", "knife"]:
                msg += f"**Weapon Detected**: {d.class_.upper()} ({d.confidence:.2f})\n"
                weapon_found = True

    if not weapon_found and (
        "CRITICAL" in incident.threat_level or "HIGH" in incident.threat_level
    ):
        if incident.label != "normal":
            msg += f"**Detected Action**: {incident.label.upper()} ({incident.confidence:.2f})\n"

    if "FIGHTING" in incident.threat_level or "VIOLENCE" in incident.threat_level:
        msg += f"**Activity**: VIOLENCE/FIGHTING ({incident.confidence:.2f})\n"

    if "ATM" in incident.threat_level:
        msg += f"**Context**: ATM Activity\n"

    msg += f"**Camera**: {incident.camera_id}\n"
    msg += f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    telegram_service.send_telegram_alert(msg, image_path)

    if "CRITICAL" in incident.threat_level or "HIGH" in incident.threat_level:
        email_service.send_email_alert(
            subject=f" Threat Alert: {incident.threat_level}",
            body=msg.replace("*", ""),  # Remove markdown for plain text email
            image_path=image_path
        )


def _notification_message(incident: schemas.IncidentCreate, created_at: datetime) -> str:
    detected_class = incident.detected_class or incident.label
    confidence = f"{incident.confidence:.1%}"
    event_time = created_at.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
    return (
        f"{incident.threat_level}: {detected_class} detected "
        f"({confidence}) at {event_time}."
    )


def _validate_threat_incident(incident: schemas.IncidentCreate) -> str | None:
    level = incident.threat_level.strip().upper()
    action = incident.data.action_classification.class_.strip().lower()
    detected_classes = {
        detection.class_.strip().lower()
        for detection in incident.data.detections
    }
    violent_action = action in {"shooting", "violence"}
    weapon_detected = bool(detected_classes.intersection({"gun", "knife"}))
    atm_detected = (
        incident.data.spatial_context.atm_present
        and "atm" in detected_classes
    )
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
    if level == "CRITICAL: ATM ROBBERY" and atm_detected and violent_action:
        return None
    if level == "WARNING: FIGHTING" and action == "violence":
        return None
    if level == "WARNING: SUSPICIOUS STANCE" and action == "shooting":
        return None
    if level == "WARNING: LOITERING AT ATM" and atm_detected:
        return None
    if (
        level == "WARNING: UNATTENDED BAGGAGE"
        and action == "unattended"
        and detected_classes.intersection({"backpack", "handbag"})
    ):
        return None
    return "Incident does not match a supported validated threat rule."


def deliver_incident_notification(
    incident_id: str,
    message: str,
    session_factory=None,
) -> None:
    db = (session_factory or database.SessionLocal)()
    try:
        ntfy_service.send_ntfy_notification(message)
    except ntfy_service.NtfyDeliveryError as exc:
        try:
            incident = db.get(models.Incident, incident_id)
            if incident is not None:
                incident.notification_status = "failed"
                incident.notification_error = str(exc)
                db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.exception(
                "Could not persist failed notification status for incident %s.",
                incident_id,
            )
        logger.error("Phone notification failed for incident %s: %s", incident_id, exc)
    else:
        try:
            incident = db.get(models.Incident, incident_id)
            if incident is not None:
                incident.notification_status = "sent"
                incident.notification_error = None
                db.commit()
        except SQLAlchemyError:
            db.rollback()
            logger.exception(
                "Phone notification succeeded but its status could not be "
                "persisted for incident %s.",
                incident_id,
            )
    finally:
        db.close()


@app.post("/api/v1/incidents", response_model=schemas.IncidentResponse)
def create_incident(
    background_tasks: BackgroundTasks,
    incident_data: str = Form(...),
    file: UploadFile = File(None),
    db: Session = Depends(database.get_db),
):
    try:
        data = json.loads(incident_data)
        incident = schemas.IncidentCreate(**data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Incident request failed schema validation: %s", exc)
        raise HTTPException(
            status_code=422,
            detail="Incident payload is invalid.",
        ) from exc
    threat_validation_error = _validate_threat_incident(incident)
    if threat_validation_error:
        raise HTTPException(status_code=422, detail=threat_validation_error)

    idempotency_key = incident.data.idempotency_key
    if idempotency_key:
        try:
            existing_incident = (
                db.query(models.Incident)
                .filter(models.Incident.idempotency_key == idempotency_key)
                .first()
            )
        except SQLAlchemyError as exc:
            logger.exception("Could not check incident idempotency.")
            raise HTTPException(
                status_code=500,
                detail="Could not check for a duplicate incident.",
            ) from exc
        if existing_incident is not None:
            logger.info(
                "Returning existing incident for duplicate idempotency key."
            )
            return existing_incident
        
    logger.info("Received incident: %s (%s)", incident.label, incident.threat_level)

    file_path = None
    if file:
        try:
            ext = os.path.splitext(file.filename or "")[1].lstrip(".") or "jpg"
            if not ext.isalnum():
                raise ValueError("Invalid snapshot file extension.")
            filename = f"{uuid.uuid4()}.{ext}"
            file_path = str(ALERT_STORAGE_DIR / filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except (OSError, ValueError):
            logger.exception("Could not save the incident snapshot.")
            raise HTTPException(
                status_code=500,
                detail="Could not save incident snapshot.",
            )

    created_at = models.utc_now()
    detected_class = incident.detected_class or incident.label
    tracking_id = incident.data.action_classification.track_id
    notification_status, notification_error = ntfy_service.notification_state()
    if incident.threat_level.upper() == "SAFE":
        notification_status = "disabled"
        notification_error = "Safe events do not generate notifications."
    elif notification_status == "ready" and incident.notification_authorized:
        notification_status = "pending"
        notification_error = None
    elif notification_status == "ready":
        notification_status = "disabled"
        notification_error = (
            "Detector validation and operator approval are required for delivery."
        )

    db_incident = models.Incident(
        created_at=created_at,
        camera_id=incident.camera_id,
        threat_level=incident.threat_level,
        label=incident.label,
        confidence=incident.confidence,
        idempotency_key=idempotency_key,
        detected_class=detected_class,
        tracking_id=str(tracking_id) if tracking_id is not None else None,
        notification_authorized=incident.notification_authorized,
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
        try:
            existing_incident = (
                db.query(models.Incident)
                .filter(models.Incident.idempotency_key == idempotency_key)
                .first()
            ) if idempotency_key else None
        except SQLAlchemyError as query_exc:
            logger.exception("Could not resolve incident idempotency conflict.")
            raise HTTPException(
                status_code=500,
                detail="Could not resolve incident write conflict.",
            ) from query_exc
        if existing_incident is None:
            logger.exception("Incident violated a database constraint.")
            raise HTTPException(
                status_code=500,
                detail="Could not save incident because of a database constraint.",
            ) from exc
        if file_path:
            try:
                os.remove(file_path)
            except OSError:
                logger.warning("Could not remove duplicate incident snapshot: %s", file_path)
        return existing_incident
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Could not persist incident.")
        if file_path:
            try:
                os.remove(file_path)
            except OSError:
                logger.warning("Could not remove unsaved incident snapshot.")
        raise HTTPException(
            status_code=500,
            detail="Could not save incident to persistent storage.",
        ) from exc
    try:
        db.refresh(db_incident)
    except SQLAlchemyError as exc:
        logger.exception("Incident was saved but could not be read back.")
        raise HTTPException(
            status_code=500,
            detail="Incident was saved but could not be read back.",
        ) from exc

    if incident.threat_level.upper() != "SAFE" and notification_status == "pending":
        background_tasks.add_task(
            deliver_incident_notification,
            db_incident.id,
            _notification_message(incident, created_at),
            sessionmaker(bind=db.get_bind()),
        )
    if (
        incident.threat_level.upper() != "SAFE"
        and incident.notification_authorized
    ):
        background_tasks.add_task(process_notifications, incident, file_path)

    return db_incident


@app.get("/api/v1/incidents", response_model=list[schemas.IncidentResponse])
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
            detail="Could not read incident history from persistent storage.",
        ) from exc


@app.get("/api/v1/incidents/summary")
def get_incident_summary(db: Session = Depends(database.get_db)):
    today = models.utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        total = db.query(func.count(models.Incident.id)).scalar() or 0
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


@app.get("/api/v1/notifications/status")
def get_notification_status():
    state, error = ntfy_service.notification_state()
    return {
        "status": state,
        "error": error,
        "enabled": settings.ALERTS_ENABLED,
    }


@app.post("/api/v1/notifications/test")
def test_notification():
    state, error = ntfy_service.notification_state()
    if state != "ready":
        raise HTTPException(status_code=409, detail=error)
    now = models.utc_now().replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
    try:
        ntfy_service.send_ntfy_notification(
            f"SmartVision test notification. Test delivery requested at {now}."
        )
    except ntfy_service.NtfyDeliveryError as exc:
        logger.error("Explicit ntfy test notification failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"status": "sent", "message": "ntfy accepted the test notification."}


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
    notification_status, notification_error = ntfy_service.notification_state()
    return {
        "status": "running",
        "database_status": database_status,
        "database_error": database_error,
        "notification_status": notification_status,
        "notification_error": notification_error,
    }
