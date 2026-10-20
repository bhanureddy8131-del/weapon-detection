import os
from pathlib import Path
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {
        "1", "true", "yes", "on"
    }


class Config:
    database_path = Path(
        os.getenv("SMARTVISION_DATABASE_PATH", ROOT_DIR / "data" / "incidents.db")
    ).expanduser()
    if not database_path.is_absolute():
        database_path = ROOT_DIR / database_path
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    DATABASE_URL = os.getenv(
        "SMARTVISION_DATABASE_URL",
        f"sqlite:///{database_path.as_posix()}",
    )

    ALERTS_ENABLED = _env_flag("SMARTVISION_ALERTS_ENABLED")
    NTFY_TOPIC = os.getenv("SMARTVISION_NTFY_TOPIC", "").strip()
    NTFY_SERVER = os.getenv("SMARTVISION_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    ALERT_APPROVERS = frozenset(
        value.strip().casefold()
        for value in os.getenv("SMARTVISION_ALERT_APPROVERS", "").split(",")
        if value.strip()
    )

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    EMAIL_FROM = os.getenv("EMAIL_FROM", "alert@smartvision.com")
    EMAIL_TO = os.getenv("EMAIL_TO", "")


settings = Config()
