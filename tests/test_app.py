"""Streamlit-level smoke tests for the explorer interface."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    def test_default_explorer_renders_without_exceptions(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, "Color spaces")
        self.assertEqual(app.selectbox[1].value, "HSV controls")
        self.assertEqual(app.segmented_control[0].label, "Appearance")
        self.assertEqual(app.segmented_control[0].value, "Light")
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Learn", "Copy the code", "Compare settings", "Histogram"],
        )
        rendered_markdown = "\n".join(block.value for block in app.markdown)
        normalized_markdown = " ".join(rendered_markdown.split())
        self.assertIn("Powered by", rendered_markdown)
        self.assertIn("Echelon Consulting", rendered_markdown)
        self.assertIn("Research mindset. Production habits.", rendered_markdown)
        self.assertIn(
            "Need help designing, implementing, improving your AI project contact us at",
            normalized_markdown,
        )
        self.assertIn("https://echelonconsulting.vercel.app", rendered_markdown)
        self.assertEqual(app.text_area[0].label, "Your feedback")
        self.assertEqual(
            app.text_input[0].label,
            "Email for a reply (optional)",
        )
        self.assertEqual(app.radio[0].label, "What would you like to share?")
        self.assertEqual(app.radio[0].value, "General feedback")
        self.assertEqual(app.button[-1].label, "Send feedback")
        self.assertNotIn("@gmail.com", rendered_markdown)

    def test_feedback_validation_keeps_an_unsent_message_private(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.text_area[0].set_value("A thoughtful feature request.")
        app.button[-1].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("not configured yet" in error.value for error in app.error))
        rendered_markdown = "\n".join(block.value for block in app.markdown)
        self.assertNotIn("@gmail.com", rendered_markdown)

    @patch.dict(
        "os.environ",
        {"FORMSPREE_ENDPOINT": "https://formspree.io/f/example-id"},
    )
    @patch("src.feedback.urlopen")
    def test_feedback_submission_shows_a_private_success_message(
        self, mock_urlopen: MagicMock
    ) -> None:
        response = MagicMock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response

        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.radio[0].set_value("Feature idea")
        app.text_area[0].set_value("Please add another threshold comparison.")
        app.text_input[0].set_value("visitor@example.com")
        app.button[-1].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("sent privately to Kasra" in success.value for success in app.success)
        )
        mock_urlopen.assert_called_once()

    def test_appearance_control_switches_to_dark_theme(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.segmented_control[0].set_value("Dark").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.segmented_control[0].value, "Dark")
        rendered_markdown = "\n".join(block.value for block in app.markdown)
        self.assertIn("--lab-bg: #0b1120", rendered_markdown)
        self.assertIn("--lab-color-scheme: dark", rendered_markdown)

    def test_switching_to_canny_updates_controls_and_code(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.selectbox[0].select("Edges").run()
        app.selectbox[1].select("Canny edges").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            [(slider.label, slider.value) for slider in app.slider],
            [("Low / high thresholds", (60, 150)), ("Pre-blur kernel", 5)],
        )
        self.assertTrue(any("cv2.Canny" in block.value for block in app.code))


if __name__ == "__main__":
    unittest.main()
