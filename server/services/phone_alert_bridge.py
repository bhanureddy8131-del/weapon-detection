"""
SmartVision Pro - Phone Alert Bridge

Connects backend incident events to ntfy phone notifications.

Environment variables:
    SMARTVISION_NTFY_ENABLED=true
    SMARTVISION_NTFY_TOPIC=your_private_topic
    SMARTVISION_ALERTS_ENABLED=true

This module handles notification delivery only.
It does not approve an unvalidated detector.
"""

import logging
import os
import time
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger("smartvision.phone_alerts")

NTFY_BASE_URL = os.getenv(
    "SMARTVISION_NTFY_BASE_URL",
    "https://ntfy.sh",
).rstrip("/")

NTFY_ENABLED = (
    os.getenv("SMARTVISION_NTFY_ENABLED", "false").lower()
    in {"1", "true", "yes", "on"}
)

ALERTS_ENABLED = (
    os.getenv("SMARTVISION_ALERTS_ENABLED", "false").lower()
    in {"1", "true", "yes", "on"}
)

NTFY_TOPIC = os.getenv("SMARTVISION_NTFY_TOPIC", "").strip()

HTTP_TIMEOUT_SECONDS = 10
MAX_ATTEMPTS = 3


class PhoneAlertError(RuntimeError):
    """Raised when a phone notification cannot be delivered."""


def notification_status() -> dict[str, Any]:
    """Return safe configuration information without exposing the topic."""

    configured = bool(NTFY_TOPIC)

    return {
        "service": "smartvision_phone_alert_bridge",
        "transport_configured": configured,
        "ntfy_enabled": NTFY_ENABLED,
        "alerts_enabled": ALERTS_ENABLED,
        "ready": configured and NTFY_ENABLED and ALERTS_ENABLED,
    }


def send_phone_alert(
    title: str,
    message: str,
    *,
    priority: str = "high",
    tags: str = "warning",
) -> dict[str, Any]:
    """
    Send a notification to the subscribed phone.

    Call this function only after the backend's existing
    notification policy authorizes the event.
    """

    if not NTFY_ENABLED:
        raise PhoneAlertError("ntfy notifications are disabled.")

    if not ALERTS_ENABLED:
        raise PhoneAlertError("Phone alerts are disabled by configuration.")

    if not NTFY_TOPIC:
        raise PhoneAlertError("SMARTVISION_NTFY_TOPIC is not configured.")

    title = str(title).strip()[:120]
    message = str(message).strip()[:1500]

    if not title or not message:
        raise PhoneAlertError("A title and message are required.")

    topic = quote(NTFY_TOPIC, safe="")
    url = f"{NTFY_BASE_URL}/{topic}"

    headers = {
        "Title": title,
        "Priority": priority,
        "Tags": tags,
    }

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                url,
                data=message.encode("utf-8"),
                headers=headers,
                timeout=HTTP_TIMEOUT_SECONDS,
            )

            if 200 <= response.status_code < 300:
                logger.info(
                    "Phone notification delivered successfully."
                )
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "attempt": attempt,
                }

            last_error = (
                f"ntfy returned HTTP {response.status_code}"
            )

            if response.status_code < 500:
                break

        except requests.RequestException as exc:
            last_error = type(exc).__name__

        if attempt < MAX_ATTEMPTS:
            time.sleep(attempt)

    logger.error(
        "Phone notification delivery failed: %s",
        last_error or "unknown error",
    )

    raise PhoneAlertError(
        f"Phone notification delivery failed: "
        f"{last_error or 'unknown error'}"
    )