"""Tests for private feedback delivery."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

from src.feedback import (
    FeedbackConfigurationError,
    FeedbackDeliveryError,
    FeedbackSubmission,
    submit_feedback,
    validate_formspree_endpoint,
)


class FeedbackDeliveryTests(unittest.TestCase):
    def test_endpoint_must_be_an_https_formspree_form(self) -> None:
        self.assertEqual(
            validate_formspree_endpoint(" https://formspree.io/f/example-id "),
            "https://formspree.io/f/example-id",
        )

        for endpoint in (
            "",
            "http://formspree.io/f/example-id",
            "https://example.com/f/example-id",
            "https://formspree.io/not-a-form",
        ):
            with (
                self.subTest(endpoint=endpoint),
                self.assertRaises(FeedbackConfigurationError),
            ):
                validate_formspree_endpoint(endpoint)

    @patch("src.feedback.urlopen")
    def test_submission_contains_feedback_context_without_a_recipient_email(
        self, mock_urlopen: MagicMock
    ) -> None:
        response = MagicMock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response

        submit_feedback(
            "https://formspree.io/f/example-id",
            FeedbackSubmission(
                category="Feature idea",
                message="Add a side-by-side threshold comparison.",
                technique="Thresholding / Otsu threshold",
                reply_email="visitor@example.com",
            ),
        )

        request = mock_urlopen.call_args.args[0]
        fields = parse_qs(request.data.decode("utf-8"))
        self.assertEqual(fields["feedback_type"], ["Feature idea"])
        self.assertEqual(fields["current_technique"], ["Thresholding / Otsu threshold"])
        self.assertEqual(fields["email"], ["visitor@example.com"])
        self.assertNotIn("recipient", fields)

    @patch("src.feedback.urlopen")
    def test_delivery_failure_is_reported(self, mock_urlopen: MagicMock) -> None:
        response = MagicMock()
        response.status = 503
        mock_urlopen.return_value.__enter__.return_value = response

        with self.assertRaises(FeedbackDeliveryError):
            submit_feedback(
                "https://formspree.io/f/example-id",
                FeedbackSubmission(
                    category="General feedback",
                    message="Useful explorer.",
                    technique="Color spaces / HSV controls",
                ),
            )
