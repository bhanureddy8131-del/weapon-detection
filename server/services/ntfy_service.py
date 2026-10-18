import logging
from urllib.parse import quote

import requests

from ..config import settings

logger = logging.getLogger(__name__)


class NtfyDeliveryError(RuntimeError):
    """A sanitized ntfy failure that never includes its private topic URL."""


def notification_state():
    if not settings.ALERTS_ENABLED:
        return "disabled", "Set SMARTVISION_ALERTS_ENABLED=true to enable phone alerts."
    if not settings.NTFY_TOPIC:
        return "misconfigured", "SMARTVISION_NTFY_TOPIC is not configured."
    return "ready", None


def _notification_url():
    state, error = notification_state()
    if state != "ready":
        raise NtfyDeliveryError(error or "ntfy is not configured.")
    return f"{settings.NTFY_SERVER}/{quote(settings.NTFY_TOPIC, safe='')}"


def send_ntfy_notification(message: str) -> None:
    url = _notification_url()
    try:
        response = requests.post(
            url,
            data=message.encode("utf-8"),
            headers={"Title": "SmartVision alert", "Tags": "warning"},
            timeout=(2.0, 5.0),
        )
        if not 200 <= response.status_code < 300:
            raise NtfyDeliveryError(
                f"ntfy returned HTTP {response.status_code}."
            )
    except requests.RequestException as exc:
        logger.error("ntfy request failed (%s).", type(exc).__name__)
        raise NtfyDeliveryError(
            f"ntfy request failed ({type(exc).__name__})."
        ) from None
