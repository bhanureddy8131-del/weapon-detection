import logging
import os
from pathlib import Path
from typing import Optional

import requests

from ..config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_TIMEOUT = (5, 15)
TELEGRAM_CAPTION_LIMIT = 1024
TELEGRAM_MESSAGE_LIMIT = 4096


def _get_telegram_credentials():
    """Return configured Telegram credentials without logging secrets."""
    token = getattr(settings, "TELEGRAM_BOT_TOKEN", None)
    chat_id = getattr(settings, "TELEGRAM_CHAT_ID", None)

    if not token or not chat_id:
        logger.warning(
            "Telegram credentials are missing. "
            "Skipping Telegram notification."
        )
        return None, None

    return str(token).strip(), str(chat_id).strip()


def _telegram_api_request(
    method: str,
    token: str,
    *,
    data: Optional[dict] = None,
    files: Optional[dict] = None,
    timeout=TELEGRAM_TIMEOUT,
):
    """
    Send a request to Telegram and safely handle API errors.

    The token is required by Telegram in the URL, so the URL is never logged.
    """
    url = f"{TELEGRAM_API_BASE}/bot{token}/{method}"

    try:
        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=timeout,
        )
    except requests.Timeout:
        logger.error(
            "Telegram %s request timed out.",
            method,
        )
        return False
    except requests.RequestException as exc:
        # Avoid logging the exception text because it may contain the URL.
        logger.error(
            "Telegram %s request failed (%s).",
            method,
            type(exc).__name__,
        )
        return False

    try:
        result = response.json()
    except ValueError:
        logger.error(
            "Telegram %s returned a non-JSON response (HTTP %s).",
            method,
            response.status_code,
        )
        return False

    if not response.ok or not result.get("ok", False):
        # Telegram's description is useful for diagnosing errors.
        # Do not log the request URL or token.
        description = result.get("description", "Unknown Telegram API error")
        error_code = result.get("error_code", response.status_code)

        logger.error(
            "Telegram %s rejected the request: HTTP %s, API error %s: %s",
            method,
            response.status_code,
            error_code,
            description,
        )
        return False

    logger.info("Telegram %s notification sent successfully.", method)
    return True


def send_telegram_alert(
    message: str,
    image_path: Optional[str] = None,
) -> bool:
    """
    Send a Telegram text alert or a photo alert.

    If a valid image path is supplied, sendPhoto is attempted.
    If the image is missing or invalid, send a text notification instead.

    Returns True when Telegram accepts the message; otherwise False.
    """
    token, chat_id = _get_telegram_credentials()

    if not token or not chat_id:
        return False

    text = str(message or "").strip()

    if not text:
        text = "SmartVision Pro: A security incident was detected."

    # Attempt to send a photo if an accessible image was supplied.
    if image_path:
        path = Path(image_path)

        if not path.is_file():
            logger.warning(
                "Telegram image is unavailable. Sending text alert instead."
            )
        else:
            try:
                with path.open("rb") as photo:
                    payload = {
                        "chat_id": chat_id,
                        "caption": text[:TELEGRAM_CAPTION_LIMIT],
                    }

                    # Do not use Markdown here: underscores, brackets,
                    # or other characters in incident text can cause
                    # Telegram entity-parsing errors.
                    files = {
                        "photo": (
                            path.name,
                            photo,
                            "application/octet-stream",
                        )
                    }

                    success = _telegram_api_request(
                        "sendPhoto",
                        token,
                        data=payload,
                        files=files,
                    )

                if success:
                    return True

                logger.warning(
                    "Telegram photo alert failed; attempting text fallback."
                )

            except OSError:
                logger.exception(
                    "Unable to read the Telegram image. "
                    "Attempting text fallback."
                )

    # Text fallback: also used when no image was supplied.
    payload = {
        "chat_id": chat_id,
        "text": text[:TELEGRAM_MESSAGE_LIMIT],
    }

    return _telegram_api_request(
        "sendMessage",
        token,
        data=payload,
        timeout=(5, 10),
    )
