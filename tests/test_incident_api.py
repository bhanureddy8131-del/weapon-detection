import json
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server import database, main, models


class IncidentEndpointTests(unittest.TestCase):
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

                listed = client.get("/api/v1/incidents", params={"limit": 5})
                self.assertEqual(listed.status_code, 200)
                records = listed.json()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["data"]["frame_id"], 3)
                self.assertIsNone(records[0]["image_path"])
        finally:
            main.app.dependency_overrides.pop(database.get_db, None)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()