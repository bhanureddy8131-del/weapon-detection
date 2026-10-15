import os
import sys

# --- PROJECT SETUP ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- PATHS ---
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolo11s_fine_tune_v3.pt")
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, "models", "lstm_action_recognition_pro_v2.pth")
POSE_MODEL_PATH = os.getenv(
    "SMARTVISION_POSE_MODEL_PATH",
    os.path.join(ROOT_DIR, "models", "pose_landmarker_lite.task"),
)
INPUT_VIDEO_DIR = os.path.join(ROOT_DIR, "data", "input_videos")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
API_BASE_URL = os.getenv(
    "SMARTVISION_API_BASE_URL",
    "http://127.0.0.1:8000/api/v1",
).rstrip("/")
SERVER_URL = f"{API_BASE_URL}/incidents"
ALERTS_ENABLED = os.getenv("SMARTVISION_ALERTS_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
ALERT_ENDPOINT_VERIFIED = os.getenv(
    "SMARTVISION_ALERT_ENDPOINT_VERIFIED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
ALERT_RETRY_ATTEMPTS = min(
    5,
    max(
        1,
    int(os.getenv("SMARTVISION_ALERT_RETRY_ATTEMPTS", "3")),
    ),
)
ALERT_RETRY_BACKOFF_SECONDS = min(
    2.0,
    max(
        0.0,
    float(os.getenv("SMARTVISION_ALERT_RETRY_BACKOFF_SECONDS", "0.5")),
    ),
)

# --- PARAMETERS ---
WEBCAM_ID = 0
CONFIDENCE_THRESHOLD = float(
    os.getenv("SMARTVISION_CONFIDENCE_THRESHOLD", "0.10")
)
YOLO_IMAGE_SIZE = int(os.getenv("SMARTVISION_YOLO_IMAGE_SIZE", "416"))
YOLO_IOU_THRESHOLD = float(os.getenv("SMARTVISION_YOLO_IOU", "0.70"))
EXPECTED_DETECTOR_CLASSES = (
    "atm",
    "backpack",
    "gun",
    "handbag",
    "knife",
    "person",
)
ACTION_CONFIDENCE_THRESHOLD = float(
    os.getenv("SMARTVISION_ACTION_CONFIDENCE_THRESHOLD", "0.85")
)
SEQUENCE_LENGTH = 30  # Must match training

DRAW_PERSON_BOXES = True # Draw bounding boxes for persons
DRAW_LABELS = True # Draw text labels (ID, Class Name)



# --- THREAT LOGIC ---
LOGGABLE_THREATS = ["violence", "shooting"]
LOG_COOLDOWN = 30.0  # Seconds between logs for the same threat type

# --- CLASS DEFINITIONS ---
CLASS_COLORS = {
    0: (128, 128, 128),  # ATM: Gray (Non-threat)
    1: (128, 128, 128),  # Backpack: Gray (Non-threat)
    2: (0, 0, 255),    # Gun: Red
    3: (128, 128, 128),  # Handbag: Gray (Non-threat)
    4: (0, 0, 255),    # Knife: Red
    5: (0, 255, 0),    # Person: Green
}
PERSON_CLASS_ID = 5
WEAPON_CLASS_IDS = [2, 4] # Gun, Knife
GUN_CLASS_IDS = [2]  # Specific IDs for firearms
KNIFE_CLASS_ID = 4
ATM_CLASS_ID = 0
BACKPACK_CLASS_ID = 1
HANDBAG_CLASS_ID = 3

CONTEXT_CLASS_IDS = [0, 1, 3] # ATM, Backpack, Handbag
CLASS_CONFIDENCE_THRESHOLDS = {
    PERSON_CLASS_ID: float(
        os.getenv("SMARTVISION_PERSON_CONFIDENCE", str(CONFIDENCE_THRESHOLD))
    ),
    GUN_CLASS_IDS[0]: float(
        os.getenv("SMARTVISION_GUN_CONFIDENCE", str(CONFIDENCE_THRESHOLD))
    ),
    KNIFE_CLASS_ID: float(
        os.getenv("SMARTVISION_KNIFE_CONFIDENCE", str(CONFIDENCE_THRESHOLD))
    ),
}

# --- TIME THRESHOLDS ---
ATM_LOITERING_THRESHOLD = 60.0  # Seconds
UNATTENDED_BAG_THRESHOLD = 60.0 # Seconds


def get_color(class_id):
    return CLASS_COLORS.get(class_id, (255, 255, 255))
