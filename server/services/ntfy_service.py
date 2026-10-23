import logging
import os
import re
import time
from threading import Lock
from urllib.parse import quote, urlparse

import requests

from ..config import settings

logger = logging.getLogger(__name__)

# HTTP client settings
CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = 8.0
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0
MAX_MESSAGE_LENGTH = 4000

# Prevent concurrent duplicate test/incident requests from
# sharing mutable notification state.
_send_lock = Lock()


class NtfyDeliveryError(RuntimeError):
    """Notification error that never exposes the private topic URL."""


def _get_server():
    """Validate and normalize the configured ntfy server."""

    server = (
        getattr(settings, "NTFY_SERVER", None)
        or "https://ntfy.sh"
    ).strip().rstrip("/")

    parsed = urlparse(server)

    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise NtfyDeliveryError(
            "Invalid ntfy server configuration. "
            "Use an HTTPS server URL."
        )

    return server


def _get_topic():
    """Read the private topic without logging or returning it."""

    topic = (
        getattr(settings, "NTFY_TOPIC", None)
        or os.getenv("SMARTVISION_NTFY_TOPIC", "")
    ).strip()

    if not topic:
        raise NtfyDeliveryError(
            "Notification topic is not configured."
        )

    # Topics must be a single path component.
    if (
        topic in {".", ".."}
        or "/" in topic
        or "\\" in topic
        or any(ord(char) < 33 for char in topic)
    ):
        raise NtfyDeliveryError(
            "The configured notification topic is invalid."
        )

    return topic


def notification_state():
    """
    Return the transport configuration state.

    This reports transport readiness only. It does not mean
    the detector has passed validation or been approved.
    """

    if not getattr(settings, "ALERTS_ENABLED", False):
        return (
            "disabled",
            "Set SMARTVISION_ALERTS_ENABLED=true to enable phone alerts.",
        )

    try:
        _get_server()
        _get_topic()

    except NtfyDeliveryError as exc:
        return "misconfigured", str(exc)

    return "ready", None


def _notification_url():
    """Build the notification URL from validated configuration."""

    state, error = notification_state()

    if state != "ready":
        raise NtfyDeliveryError(
            error or "Notification transport is not ready."
        )

    server = _get_server()
    topic = _get_topic()

    return f"{server}/{quote(topic, safe='')}"


def send_ntfy_notification(
    message: str,
    *,
    title: str = "SmartVision alert",
    priority: str = "high",
    tags: str = "warning",
) -> None:
    """
    Send a phone notification through ntfy.

    Retries transient network/server failures. Never logs the
    private topic, complete request URL, or message contents.

    This function handles delivery only; server.main remains
    responsible for deciding whether an incident is authorized.
    """

    if not isinstance(message, str) or not message.strip():
        raise NtfyDeliveryError(
            "Notification message cannot be empty."
        )

    message = message.strip()

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH - 20] + " [truncated]"

    allowed_priorities = {
        "min",
        "low",
        "default",
        "high",
        "urgent",
        "1",
        "2",
        "3",
        "4",
        "5",
    }

    if priority not in allowed_priorities:
        raise NtfyDeliveryError(
            "Invalid notification priority."
        )

    # Prevent CR/LF header injection.
    if (
        not isinstance(title, str)
        or "\r" in title
        or "\n" in title
    ):
        raise NtfyDeliveryError(
            "Invalid notification title."
        )

    if (
        not isinstance(tags, str)
        or not re.fullmatch(r"[A-Za-z0-9_,+-]{0,100}", tags)
    ):
        raise NtfyDeliveryError(
            "Invalid notification tags."
        )

    url = _notification_url()

    headers = {
        "Title": title[:100],
        "Priority": priority,
        "Tags": tags,
        "Content-Type": "text/plain; charset=utf-8",
    }

    last_error = "Notification delivery failed."

    with _send_lock:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = requests.post(
                    url,
                    data=message.encode("utf-8"),
                    headers=headers,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )

                if 200 <= response.status_code < 300:
                    logger.info(
                        "ntfy accepted notification (attempt %s).",
                        attempt,
                    )
                    return

                # Authentication, malformed requests, and other
                # non-transient client errors should not be retried.
                if 400 <= response.status_code < 500:
                    logger.error(
                        "ntfy rejected notification with HTTP %s.",
                        response.status_code,
                    )

                    raise NtfyDeliveryError(
                        f"ntfy rejected the notification "
                        f"(HTTP {response.status_code})."
                    )

                last_error = (
                    f"ntfy returned HTTP {response.status_code}."
                )

                logger.warning(
                    "ntfy server error on attempt %s: HTTP %s.",
                    attempt,
                    response.status_code,
                )

            except requests.Timeout:
                last_error = "ntfy request timed out."

                logger.warning(
                    "ntfy request timed out on attempt %s.",
                    attempt,
                )

            except requests.ConnectionError:
                last_error = "Could not connect to the ntfy server."

                logger.warning(
                    "ntfy connection failed on attempt %s.",
                    attempt,
                )

            except requests.RequestException as exc:
                logger.error(
                    "ntfy request failed (%s).",
                    type(exc).__name__,
                )

                raise NtfyDeliveryError(
                    "ntfy request failed. Check the server connection."
                ) from None

            finally:
                if "response" in locals():
                    response.close()
                    del response

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    raise NtfyDeliveryError(last_error)

