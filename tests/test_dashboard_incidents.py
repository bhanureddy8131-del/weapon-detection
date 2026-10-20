import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
from streamlit.testing.v1 import AppTest

from pipeline.app import (
    build_validation_acceptance_table,
    build_recent_detection_table,
    fetch_incident_summary,
    fetch_incidents,
    normalize_incidents,
)


class DashboardIncidentTests(unittest.TestCase):
    def test_detection_table_keeps_valid_inference_predictions(self):
        table = build_recent_detection_table([{
            "timestamp": "2026-10-17 12:30:00",
            "class": "gun",
            "confidence": 0.91,
            "track_id": 7,
            "bbox": [1, 2, 3, 4],
        }])

        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["class"], "gun")
        self.assertEqual(table.iloc[0]["confidence"], 0.91)
        self.assertEqual(
            list(table.columns),
            ["timestamp", "class", "confidence", "track_id"],
        )

    def test_malformed_detection_history_is_empty_not_sampled(self):
        table = build_recent_detection_table({"class": "gun"})

        self.assertTrue(table.empty)

    def test_acceptance_table_formats_class_list_and_is_arrow_compatible(self):
        table = build_validation_acceptance_table({
            "representative_tests_complete": {
                "passed": True,
                "actual": ["atm", "backpack", "gun", "handbag", "knife", "person"],
                "expected": ["atm", "backpack", "gun", "handbag", "knife", "person"],
            },
            "precision": {
                "passed": False,
                "actual": 0.1214,
                "minimum": 0.7,
            },
        })

        expected_classes = "atm, backpack, gun, handbag, knife, person"
        self.assertEqual(table.iloc[0]["Actual"], expected_classes)
        self.assertEqual(table.iloc[0]["Required"], expected_classes)
        self.assertEqual(table.iloc[1]["Actual"], "0.1214")
        self.assertEqual(str(table["Actual"].dtype), "string")
        self.assertEqual(str(table["Required"].dtype), "string")

        arrow_table = pa.Table.from_pandas(table, preserve_index=False)
        self.assertEqual(arrow_table.num_rows, 2)

        app = AppTest.from_string(
            "import pandas as pd\n"
            "import streamlit as st\n"
            "st.dataframe(pd.DataFrame(" + repr(table.to_dict("records")) + "))"
        )
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)

    def test_acceptance_table_handles_empty_checks_and_missing_values(self):
        empty = build_validation_acceptance_table({})
        self.assertTrue(empty.empty)
        self.assertEqual(str(empty["Actual"].dtype), "string")
        self.assertEqual(str(empty["Passed"].dtype), "bool")
        pa.Table.from_pandas(empty, preserve_index=False)

        table = build_validation_acceptance_table({
            "missing_actual": {"passed": False},
        })
        self.assertEqual(table.iloc[0]["Actual"], "Unavailable")
        self.assertEqual(table.iloc[0]["Required"], "Unavailable")
        pa.Table.from_pandas(table, preserve_index=False)

    def test_detection_size_widget_preserves_session_state_on_rerun(self):
        app = AppTest.from_string(
            "from types import SimpleNamespace\n"
            "from pipeline.app import render_detection_settings\n"
            "runtime = {\n"
            "    'detector': SimpleNamespace(\n"
            "        class_names={0: 'atm', 1: 'backpack', 2: 'gun', "
            "3: 'handbag', 4: 'knife', 5: 'person'},\n"
            "        is_custom_model=True,\n"
            "    ),\n"
            "    'threat_manager': SimpleNamespace(log_cooldown=30.0),\n"
            "}\n"
            "render_detection_settings(runtime)\n"
        ).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["smartvision_yolo_input_size"], 320)

        app.session_state["smartvision_yolo_input_size"] = 640
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["smartvision_yolo_input_size"], 640)

    def test_summary_fetch_uses_incident_summary_url_and_validates_shape(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "total": 3,
            "today": 1,
            "by_class": {"GUN": 2, "KNIFE": 1},
        }
        with patch("pipeline.app.requests.get", return_value=response) as get:
            summary, error = fetch_incident_summary()

        self.assertIsNone(error)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["by_class"]["GUN"], 2)
        self.assertEqual(
            get.call_args.args[0],
            "http://127.0.0.1:8000/api/v1/incidents/summary",
        )

    def test_malformed_summary_is_reported_not_replaced_with_zeroes(self):
        response = Mock(status_code=200)
        response.json.return_value = {"total": "many", "today": 0}
        with patch("pipeline.app.requests.get", return_value=response):
            summary, error = fetch_incident_summary()

        self.assertIsNone(summary)
        self.assertIn("malformed", error)

    def test_saved_incident_fields_are_ready_for_history_display(self):
        incident = {
            "id": "persisted-incident-id",
            "created_at": "2026-10-17T12:30:00",
            "threat_level": "HIGH: WEAPON DETECTED",
            "label": "GUN DETECTION",
            "detected_class": "GUN",
            "confidence": 0.91,
            "tracking_id": "7",
            "notification_status": "failed",
            "notification_error": "ntfy returned HTTP 503.",
            "camera_id": "test-camera",
        }
        frame, invalid_count = normalize_incidents([incident])

        self.assertEqual(invalid_count, 0)
        self.assertEqual(frame.iloc[0]["detected_class"], "GUN")
        self.assertEqual(frame.iloc[0]["confidence"], 0.91)
        self.assertEqual(frame.iloc[0]["tracking_id"], "7")
        self.assertEqual(frame.iloc[0]["notification_status"], "failed")
        self.assertEqual(
            frame.iloc[0]["notification_error"],
            "ntfy returned HTTP 503.",
        )

    def test_history_fetch_is_not_cached_between_refreshes(self):
        first = Mock(status_code=200)
        first.json.return_value = []
        refreshed = Mock(status_code=200)
        refreshed.json.return_value = [{"id": "saved-after-refresh"}]
        with patch(
            "pipeline.app.requests.get",
            side_effect=[first, refreshed],
        ) as get:
            before_refresh, first_error = fetch_incidents()
            after_refresh, second_error = fetch_incidents()

        self.assertIsNone(first_error)
        self.assertIsNone(second_error)
        self.assertEqual(before_refresh, [])
        self.assertEqual(after_refresh[0]["id"], "saved-after-refresh")
        self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
