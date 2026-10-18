from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import settings

connect_args = (
    {"check_same_thread": False}
    if settings.DATABASE_URL.startswith("sqlite")
    else {}
)
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_incident_columns():
    """Add alert fields to databases created by earlier application versions."""
    inspector = inspect(engine)
    if "incidents" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("incidents")}
    additions = {
        "idempotency_key": "VARCHAR",
        "detected_class": "VARCHAR",
        "tracking_id": "VARCHAR",
        "notification_authorized": "BOOLEAN NOT NULL DEFAULT 0",
        "notification_status": "VARCHAR NOT NULL DEFAULT 'disabled'",
        "notification_error": "TEXT",
    }
    with engine.begin() as connection:
        for name, column_type in additions.items():
            if name not in columns:
                connection.execute(
                    text(f"ALTER TABLE incidents ADD COLUMN {name} {column_type}")
                )
        connection.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "ux_incidents_idempotency_key "
            "ON incidents (idempotency_key)"
        ))


def ensure_incident_idempotency_column():
    """Compatibility wrapper for older callers."""
    ensure_incident_columns()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
