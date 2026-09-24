"""Streamlit-level smoke tests for the explorer interface."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    def test_default_explorer_renders_without_exceptions(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, "Color spaces")
        self.assertEqual(app.selectbox[1].value, "HSV controls")
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Learn", "Copy the code", "Compare settings", "Histogram"],
        )

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
