import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, String, Float, DateTime, Text, JSON
from .database import Base


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=utc_now, nullable=False)
    camera_id = Column(String, default="cam_01")
    threat_level = Column(String, nullable=False)
    label = Column(String, nullable=False)
    confidence = Column(Float)
    image_path = Column(String)
    idempotency_key = Column(String, unique=True, index=True, nullable=True)
    detected_class = Column(String)
    tracking_id = Column(String)
    notification_authorized = Column(Boolean, nullable=False, default=False)
    notification_status = Column(String, nullable=False, default="disabled")
    notification_error = Column(Text)
    data = Column(JSON)
