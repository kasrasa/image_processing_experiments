"""Small regression suite covering every registered operation."""

from __future__ import annotations

import unittest

import numpy as np

from src.image_utils import create_sample_image
from src.operation_registry import OPERATION_LIST, code_snippet, default_parameters
from src.operations import add_gaussian_noise, adjust_hsv, apply_operation


class OperationSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = create_sample_image(width=320, height=220)

    def test_every_registered_operation_returns_displayable_uint8_image(self) -> None:
        for operation in OPERATION_LIST:
            with self.subTest(operation=operation.key):
                result = apply_operation(
                    operation.key,
                    self.image,
                    default_parameters(operation),
                )
                self.assertEqual(result.image.dtype, np.uint8)
                self.assertIn(result.image.ndim, (2, 3))
                self.assertGreater(result.image.shape[0], 0)
                self.assertGreater(result.image.shape[1], 0)
                if result.image.ndim == 3:
                    self.assertEqual(result.image.shape[2], 3)

    def test_hsv_scaling_clips_instead_of_wrapping(self) -> None:
        bright = np.full((8, 8, 3), 250, dtype=np.uint8)
        result = adjust_hsv(bright, hue_shift=0, saturation_scale=1.0, value_scale=1.75)
        self.assertEqual(int(result.min()), 255)
        self.assertEqual(int(result.max()), 255)

    def test_noise_preview_is_reproducible(self) -> None:
        first = add_gaussian_noise(self.image, sigma=20, noise_mode="Color")
        second = add_gaussian_noise(self.image, sigma=20, noise_mode="Color")
        np.testing.assert_array_equal(first, second)

    def test_copyable_code_matches_each_default_preview(self) -> None:
        for operation in OPERATION_LIST:
            with self.subTest(operation=operation.key):
                parameters = default_parameters(operation)
                expected = apply_operation(operation.key, self.image, parameters).image
                namespace = {"image_rgb": self.image.copy()}
                exec(code_snippet(operation.key, parameters), namespace)  # noqa: S102
                np.testing.assert_array_equal(namespace["result"], expected)

    def test_copyable_code_matches_alternate_branches(self) -> None:
        cases = (
            ("sobel_edges", {"kernel_size": 5, "direction": "Horizontal changes (dx)"}),
            (
                "clahe",
                {"clip_limit": 3.5, "grid_size": 6, "output_mode": "Grayscale"},
            ),
            ("resize", {"scale": 1.2, "interpolation": "Lanczos"}),
            ("rotate", {"angle": -32.0, "scale": 0.9, "expand_canvas": False}),
            (
                "cutout",
                {
                    "size_ratio": 0.35,
                    "location": "Random (fixed seed)",
                    "fill": "White",
                },
            ),
            ("gaussian_noise", {"sigma": 14.0, "noise_mode": "Monochrome"}),
        )

        for key, parameters in cases:
            with self.subTest(operation=key):
                expected = apply_operation(key, self.image, parameters).image
                namespace = {"image_rgb": self.image.copy()}
                exec(code_snippet(key, parameters), namespace)  # noqa: S102
                np.testing.assert_array_equal(namespace["result"], expected)


if __name__ == "__main__":
    unittest.main()
