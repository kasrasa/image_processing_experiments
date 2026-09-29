"""Tests for uploaded-image and input-mode helpers."""

from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from src.image_utils import (
    convert_to_grayscale_rgb,
    is_grayscale_image,
    load_uploaded_image,
)


class ImageUtilityTests(unittest.TestCase):
    def test_grayscale_upload_is_accepted_as_rgb_compatible_input(self) -> None:
        uploaded = BytesIO()
        Image.fromarray(
            np.array([[0, 64], [128, 255]], dtype=np.uint8)
        ).save(uploaded, format="PNG")

        decoded = load_uploaded_image(uploaded)

        self.assertEqual(decoded.shape, (2, 2, 3))
        self.assertTrue(is_grayscale_image(decoded))

    def test_grayscale_mode_preserves_shape_without_mutating_color_source(self) -> None:
        color_source = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [90, 120, 150]]],
            dtype=np.uint8,
        )
        original = color_source.copy()

        grayscale = convert_to_grayscale_rgb(color_source)

        self.assertEqual(grayscale.shape, color_source.shape)
        self.assertTrue(is_grayscale_image(grayscale))
        np.testing.assert_array_equal(color_source, original)

    def test_color_image_is_not_reported_as_grayscale(self) -> None:
        color = np.array([[[20, 40, 60]]], dtype=np.uint8)

        self.assertFalse(is_grayscale_image(color))


if __name__ == "__main__":
    unittest.main()
