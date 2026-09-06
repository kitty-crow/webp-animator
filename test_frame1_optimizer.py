from __future__ import annotations

import time
import unittest

from PIL import Image, ImageDraw

import frame1_optimizer


class Frame1OptimizerTests(unittest.TestCase):
    def _anchor(self, width=640, height=480):
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((120, 70, 500, 410), radius=48, fill=(230, 80, 40, 255))
        draw.ellipse((205, 130, 335, 260), fill=(30, 180, 240, 255))
        draw.rectangle((365, 250, 460, 365), fill=(250, 230, 60, 255))
        return image

    def test_recovers_uniform_scale_quickly(self):
        anchor = self._anchor()
        source = anchor.resize((704, 528), Image.Resampling.LANCZOS)

        started = time.perf_counter()
        transformed, scale, shift = frame1_optimizer.find_best_scale_and_translation(
            anchor,
            source,
            axis="xy",
            max_shift=64,
            sigma=24.0,
            alpha_threshold=8,
        )
        elapsed = time.perf_counter() - started

        expected = 640 / 704
        self.assertAlmostEqual(scale, expected, delta=0.035)
        self.assertLessEqual(abs(transformed.width - 640), 10)
        self.assertLessEqual(abs(transformed.height - 480), 10)
        self.assertLessEqual(abs(shift.dx), 3)
        self.assertLessEqual(abs(shift.dy), 3)
        self.assertLess(
            elapsed,
            5.0,
            "Frame-1 scale/pan matching regressed into a long-running exhaustive search.",
        )

    def test_axis_none_keeps_translation_zero(self):
        anchor = self._anchor(320, 240)
        source = anchor.resize((352, 264), Image.Resampling.LANCZOS)
        _, _, shift = frame1_optimizer.find_best_scale_and_translation(
            anchor,
            source,
            axis="none",
            max_shift=200,
            sigma=24.0,
            alpha_threshold=8,
        )
        self.assertEqual((shift.dx, shift.dy), (0, 0))


if __name__ == "__main__":
    unittest.main()
