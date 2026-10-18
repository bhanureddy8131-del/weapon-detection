import json
import tempfile
import unittest
from pathlib import Path
from contextlib import contextmanager

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch

from server import database, main, models
from server.config import settings
from server.services.ntfy_service import NtfyDeliveryError


@contextmanager
def client_for_engine(engine):
    models.Base.metadata.create_all(bind=engine)
    test_session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    def override_get_db():
        session = test_session_factory()
        try:
            yield session
        finally:
            session.close()

    main.app.dependency_overrides[database.get_db] = override_get_db
    try:
        with TestClient(main.app) as client:
            yield client, test_session_factory
    finally:
        main.app.dependency_overrides.pop(database.get_db, None)


def create_test_engine():
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def threat_payload(idempotency_key="test-event-key"):
    return {
        "camera_id": "test-camera",
        "threat_level": "HIGH: WEAPON DETECTED",
        "label": "GUN DETECTION",
        "confidence": 0.91,
        "detected_class": "GUN",
        "notification_authorized": True,
        "data": {
            "frame_id": 3,
            "detections": [
                {
                    "class": "gun",
                    "confidence": 0.91,
                    "bbox": [10, 12, 40, 52],
                }
            ],
            "action_classification": {
                "class": "gun detection",
                "confidence": 0.91,
                "track_id": 7,
            },
            "spatial_context": {"atm_present": False, "near_atm": False},
            "idempotency_key": idempotency_key,
        },
    }


class IncidentEndpointTests(unittest.TestCase):
    def test_existing_sqlite_incident_table_gets_idempotency_column(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        with engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE incidents (id VARCHAR PRIMARY KEY)"
            ))

        try:
            with patch.object(database, "engine", engine):
                database.ensure_incident_idempotency_column()

            columns = {
                column["name"]
                for column in inspect(engine).get_columns("incidents")
            }
            indexes = inspect(engine).get_indexes("incidents")
            self.assertIn("idempotency_key", columns)
            self.assertIn("detected_class", columns)
            self.assertIn("tracking_id", columns)
            self.assertIn("notification_authorized", columns)
            self.assertIn("notification_status", columns)
            self.assertIn("notification_error", columns)
            self.assertIn(
                "ux_incidents_idempotency_key",
                {index["name"] for index in indexes},
            )
        finally:
            engine.dispose()

    def test_multipart_post_without_snapshot_matches_backend_schema(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        models.Base.metadata.create_all(bind=engine)
        test_session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
        )

        def override_get_db():
            session = test_session_factory()
            try:
                yield session
            finally:
                session.close()

        main.app.dependency_overrides[database.get_db] = override_get_db
        payload = {
            "camera_id": "test-camera",
            "threat_level": "SAFE",
            "label": "TEST PAYLOAD",
            "confidence": 0.0,
            "data": {
                "frame_id": 3,
                "event_type": "object_detection",
                "detections": [
                    {
                        "class": "knife detection",
                        "confidence": 0.91,
                        "bbox": [10, 12, 40, 52],
                    }
                ],
                "action_classification": {
                    "class": "knife detection",
                    "confidence": 0.91,
                    "track_id": 7,
                },
                "spatial_context": {
                    "atm_present": False,
                    "near_atm": False,
                },
                "idempotency_key": "test-event-key",
            },
        }

        try:
            with TestClient(main.app) as client:
                response = client.post(
                    "/api/v1/incidents",
                    data={"incident_data": json.dumps(payload)},
                    files={"_multipart": ("", b"", "text/plain")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                created = response.json()
                self.assertEqual(created["camera_id"], "test-camera")
                self.assertIsNone(created["image_path"])

                duplicate = client.post(
                    "/api/v1/incidents",
                    data={"incident_data": json.dumps(payload)},
                    files={"_multipart": ("", b"", "text/plain")},
                )
                self.assertEqual(duplicate.status_code, 200, duplicate.text)
                self.assertEqual(duplicate.json()["id"], created["id"])

                listed = client.get("/api/v1/incidents", params={"limit": 5})
                self.assertEqual(listed.status_code, 200)
                records = listed.json()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["data"]["frame_id"], 3)
                self.assertIsNone(records[0]["image_path"])
        finally:
            main.app.dependency_overrides.pop(database.get_db, None)
            engine.dispose()

    def test_incident_is_persisted_with_detection_and_notification_fields(self):
        engine = create_test_engine()
        try:
            with (
                client_for_engine(engine) as (client, _),
                patch.object(settings, "ALERTS_ENABLED", True),
                patch.object(settings, "NTFY_TOPIC", "unit-test-topic"),
                patch(
                    "server.main.ntfy_service.send_ntfy_notification"
                ) as send_notification,
            ):
                response = client.post(
                    "/api/v1/incidents",
                    data={"incident_data": json.dumps(threat_payload())},
                    files={"_multipart": ("", b"", "text/plain")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                created = response.json()
                self.assertEqual(created["detected_class"], "GUN")
                self.assertEqual(created["tracking_id"], "7")
                send_notification.assert_called_once()
                message = send_notification.call_args.args[0]
                self.assertIn("GUN", message)
                self.assertIn("91.0%", message)

                listed = client.get("/api/v1/incidents")
                self.assertEqual(listed.status_code, 200)
                self.assertEqual(len(listed.json()), 1)
                self.assertEqual(listed.json()[0]["id"], created["id"])
                self.assertEqual(listed.json()[0]["notification_status"], "sent")
                summary = client.get("/api/v1/incidents/summary")
                self.assertEqual(
                    summary.json(),
                    {"total": 1, "today": 1, "by_class": {"GUN": 1}},
                )
        finally:
            engine.dispose()

    def test_summary_for_empty_database_has_zero_counts(self):
        engine = create_test_engine()
        try:
            with client_for_engine(engine) as (client, _):
                response = client.get("/api/v1/incidents/summary")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {"total": 0, "today": 0, "by_class": {}},
            )
        finally:
            engine.dispose()

    def test_summary_route_is_registered_once_at_expected_prefix(self):
        routes = {
            getattr(route, "path", None)
            for route in main.app.routes
        }
        self.assertIn("/api/v1/incidents/summary", routes)
        self.assertNotIn("/api/v1/api/v1/incidents/summary", routes)
        self.assertIn("/api/v1/incidents/summary", main.app.openapi()["paths"])

    def test_ntfy_failure_does_not_remove_saved_incident(self):
        engine = create_test_engine()
        try:
            with (
                client_for_engine(engine) as (client, _),
                patch.object(settings, "ALERTS_ENABLED", True),
                patch.object(settings, "NTFY_TOPIC", "unit-test-topic"),
                patch(
                    "server.main.ntfy_service.send_ntfy_notification",
                    side_effect=NtfyDeliveryError("ntfy returned HTTP 503."),
                ),
            ):
                response = client.post(
                    "/api/v1/incidents",
                    data={"incident_data": json.dumps(threat_payload())},
                    files={"_multipart": ("", b"", "text/plain")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                incident_id = response.json()["id"]
                history = client.get("/api/v1/incidents").json()
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]["id"], incident_id)
                self.assertEqual(history[0]["notification_status"], "failed")
                self.assertEqual(
                    history[0]["notification_error"],
                    "ntfy returned HTTP 503.",
                )
        finally:
            engine.dispose()

    def test_unapproved_detector_cannot_trigger_phone_notification(self):
        engine = create_test_engine()
        payload = threat_payload()
        payload["notification_authorized"] = False
        try:
            with (
                client_for_engine(engine) as (client, _),
                patch.object(settings, "ALERTS_ENABLED", True),
                patch.object(settings, "NTFY_TOPIC", "unit-test-topic"),
                patch(
                    "server.main.ntfy_service.send_ntfy_notification"
                ) as send_notification,
            ):
                response = client.post(
                    "/api/v1/incidents",
                    data={"incident_data": json.dumps(payload)},
                    files={"_multipart": ("", b"", "text/plain")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse(response.json()["notification_authorized"])
                self.assertEqual(
                    response.json()["notification_status"],
                    "disabled",
                )
                history = client.get("/api/v1/incidents").json()
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]["notification_status"], "disabled")
                send_notification.assert_not_called()
        finally:
            engine.dispose()

    def test_idempotency_prevents_duplicate_incidents_and_notifications(self):
        engine = create_test_engine()
        try:
            with (
                client_for_engine(engine) as (client, _),
                patch.object(settings, "ALERTS_ENABLED", True),
                patch.object(settings, "NTFY_TOPIC", "unit-test-topic"),
                patch(
                    "server.main.ntfy_service.send_ntfy_notification"
                ) as send_notification,
            ):
                first = client.post(
                    "/api/v1/incidents",
                    data={"incident_data": json.dumps(threat_payload())},
                    files={"_multipart": ("", b"", "text/plain")},
                )
                duplicate = client.post(
                    "/api/v1/incidents",
                    data={"incident_data": json.dumps(threat_payload())},
                    files={"_multipart": ("", b"", "text/plain")},
                )
                self.assertEqual(first.status_code, 200, first.text)
                self.assertEqual(duplicate.status_code, 200, duplicate.text)
                self.assertEqual(first.json()["id"], duplicate.json()["id"])
                self.assertEqual(len(client.get("/api/v1/incidents").json()), 1)
                send_notification.assert_called_once()
        finally:
            engine.dispose()

    def test_unrelated_object_is_not_accepted_as_a_weapon_threat(self):
        engine = create_test_engine()
        payload = threat_payload()
        payload["data"]["detections"][0]["class"] = "backpack"
        payload["detected_class"] = "BACKPACK"
        payload["label"] = "BACKPACK DETECTION"
        try:
            with (
                client_for_engine(engine) as (client, _),
                patch.object(settings, "ALERTS_ENABLED", True),
                patch.object(settings, "NTFY_TOPIC", "unit-test-topic"),
                patch(
                    "server.main.ntfy_service.send_ntfy_notification"
                ) as send_notification,
            ):
                response = client.post(
                    "/api/v1/incidents",
                    data={"incident_data": json.dumps(payload)},
                    files={"_multipart": ("", b"", "text/plain")},
                )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(client.get("/api/v1/incidents").json(), [])
                send_notification.assert_not_called()
        finally:
            engine.dispose()

    def test_test_notification_is_safe_and_does_not_create_an_incident(self):
        engine = create_test_engine()
        try:
            with (
                client_for_engine(engine) as (client, _),
                patch.object(settings, "ALERTS_ENABLED", False),
                patch.object(settings, "NTFY_TOPIC", ""),
                patch(
                    "server.main.ntfy_service.send_ntfy_notification"
                ) as send_notification,
            ):
                disabled = client.post("/api/v1/notifications/test")
                self.assertEqual(disabled.status_code, 409)
                send_notification.assert_not_called()

            with (
                client_for_engine(engine) as (client, _),
                patch.object(settings, "ALERTS_ENABLED", True),
                patch.object(settings, "NTFY_TOPIC", "unit-test-topic"),
                patch(
                    "server.main.ntfy_service.send_ntfy_notification"
                ) as send_notification,
            ):
                response = client.post("/api/v1/notifications/test")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["status"], "sent")
                self.assertEqual(client.get("/api/v1/incidents").json(), [])
                send_notification.assert_called_once()
        finally:
            engine.dispose()

    def test_incident_history_survives_database_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incidents.db"
            url = f"sqlite:///{path.as_posix()}"
            first_engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
            )
            with (
                client_for_engine(first_engine) as (client, _),
                patch.object(settings, "ALERTS_ENABLED", False),
                patch.object(settings, "NTFY_TOPIC", ""),
            ):
                response = client.post(
                    "/api/v1/incidents",
                    data={"incident_data": json.dumps(threat_payload())},
                    files={"_multipart": ("", b"", "text/plain")},
                )
                self.assertEqual(response.status_code, 200, response.text)
                incident_id = response.json()["id"]
            first_engine.dispose()

            reopened_engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
            )
            try:
                with client_for_engine(reopened_engine) as (client, _):
                    history = client.get("/api/v1/incidents")
                    self.assertEqual(history.status_code, 200)
                    self.assertEqual(history.json()[0]["id"], incident_id)
            finally:
                reopened_engine.dispose()

    def test_threat_manager_suppresses_repeated_tracked_incidents(self):
        from src.core.threat_manager import ThreatManager

        manager = ThreatManager(alerts_enabled=False)
        try:
            with patch.object(manager, "_send_alert_async", return_value={"id": "queued"}) as send:
                for _ in range(2):
                    manager.log_threat(
                        track_id=21,
                        source_name="test-camera",
                        action_label="gun detection",
                        action_prob=0.91,
                        threat_level="HIGH: WEAPON DETECTED",
                    )
            send.assert_called_once()
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()