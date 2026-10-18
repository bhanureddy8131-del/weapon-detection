from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from datetime import datetime

class Detection(BaseModel):
    class_: str = Field(..., alias="class")
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[int]


class ActionClassification(BaseModel):
    class_: str = Field(..., alias="class")
    confidence: float = Field(ge=0.0, le=1.0)
    track_id: Optional[int] = None


class SpatialContext(BaseModel):
    atm_present: bool = False
    near_atm: bool = False


class IncidentData(BaseModel):
    frame_id: int
    detections: List[Detection]
    action_classification: ActionClassification
    spatial_context: SpatialContext
    idempotency_key: Optional[str] = None


class IncidentCreate(BaseModel):
    camera_id: str = "cam_01"
    threat_level: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    detected_class: Optional[str] = None
    notification_authorized: bool = False
    data: IncidentData


class IncidentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    camera_id: str
    threat_level: str
    label: str
    confidence: float
    detected_class: Optional[str] = None
    tracking_id: Optional[str] = None
    notification_authorized: bool
    notification_status: str
    notification_error: Optional[str] = None
    image_path: Optional[str] = None
    data: IncidentData
