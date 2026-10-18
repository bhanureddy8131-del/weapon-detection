import unittest
from unittest.mock import Mock, patch

import requests

from server.config import settings
from server.services.ntfy_service import (
    NtfyDeliveryError,
    send_ntfy_notification,
)


class NtfyServiceTests(unittest.TestCase):
    def test_success_requires_a_successful_http_response(self):
        response = Mock(status_code=202)
        with (
            patch.object(settings, "ALERTS_ENABLED", True),
            patch.object(settings, "NTFY_TOPIC", "unit-test-topic"),
            patch("server.services.ntfy_service.requests.post", return_value=response) as post,
        ):
            send_ntfy_notification("Safe test")

        self.assertTrue(post.call_args.kwargs["timeout"])
        self.assertIn("unit-test-topic", post.call_args.args[0])

    def test_non_success_response_is_not_reported_as_delivered(self):
        response = Mock(status_code=503)
        with (
            patch.object(settings, "ALERTS_ENABLED", True),
            patch.object(settings, "NTFY_TOPIC", "private-test-topic"),
            patch("server.services.ntfy_service.requests.post", return_value=response),
        ):
            with self.assertRaises(NtfyDeliveryError) as raised:
                send_ntfy_notification("Safe test")

        self.assertIn("503", str(raised.exception))
        self.assertNotIn("private-test-topic", str(raised.exception))

    def test_connection_failure_is_sanitized(self):
        with (
            patch.object(settings, "ALERTS_ENABLED", True),
            patch.object(settings, "NTFY_TOPIC", "private-test-topic"),
            patch(
                "server.services.ntfy_service.requests.post",
                side_effect=requests.ConnectionError(
                    "failed to post https://ntfy.sh/private-test-topic"
                ),
            ),
        ):
            with self.assertRaises(NtfyDeliveryError) as raised:
                send_ntfy_notification("Safe test")

        self.assertNotIn("private-test-topic", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
