import os
import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))


def send_ntfy_message(message, title="SmartVision Pro", priority=3):
    enabled = os.getenv("SMARTVISION_NTFY_ENABLED", "false").lower() == "true"
    topic = os.getenv("SMARTVISION_NTFY_TOPIC", "").strip()
    server = os.getenv("SMARTVISION_NTFY_SERVER", "https://ntfy.sh").rstrip("/")

    if not enabled:
        return False, "ntfy integration is disabled"

    if not topic or topic == "PASTE_YOUR_TOPIC_HERE":
        return False, "Set SMARTVISION_NTFY_TOPIC in the project .env file"

    if not all(c.isalnum() or c in "-_" for c in topic):
        return False, "Invalid ntfy topic name"

    try:
        response = requests.post(
            f"{server}/{topic}",
            data=str(message).encode("utf-8"),
            headers={
                "Title": title,
                "Priority": str(max(1, min(5, int(priority)))),
                "Tags": "bell",
            },
            timeout=10,
        )
        response.raise_for_status()
        return True, "ntfy notification sent"
    except requests.RequestException as exc:
        return False, f"ntfy delivery failed: {exc}"
