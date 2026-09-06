from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from webp_fast import _prepare_delta_frames, save_webp_fast


class FastWebPTests(unittest.TestCase):
    def _frames(self):
        frames = []
        first = Image.new("RGBA", (40, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(first)
        draw.rectangle((4, 8, 12, 18), fill=(240, 40, 30, 255))
        frames.append(first)

        second = Image.new("RGBA", first.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(second)
        draw.rectangle((9, 8, 17, 18), fill=(240, 40, 30, 255))
        frames.append(second)

        third = second.copy()
        draw = ImageDraw.Draw(third)
        draw.rectangle((25, 4, 29, 8), fill=(30, 120, 250, 255))
        frames.append(third)
        return frames

    def test_lossless_delta_round_trip_is_pixel_exact(self):
        frames = self._frames()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "animation.webp"
            save_webp_fast(
                frames,
                output,
                durations=[100, 65, 135],
                loop=0,
                lossless=True,
                quality=90,
            )
            with Image.open(output) as decoded:
                self.assertEqual(decoded.format, "WEBP")
                self.assertEqual(decoded.n_frames, 3)
                for index, expected in enumerate(frames):
                    decoded.seek(index)
                    actual = decoded.convert("RGBA")
                    difference = ImageChops.difference(expected, actual)
                    self.assertFalse(
                        any(channel.getbbox() for channel in difference.split()),
                        f"frame {index} changed during lossless delta encoding",
                    )

    def test_delta_preparation_uses_small_rectangles_and_safe_blending(self):
        frames = self._frames()
        prepared = _prepare_delta_frames(frames, [100, 100, 100])
        self.assertEqual(len(prepared), 3)
        self.assertEqual(prepared[0].image.size, frames[0].size)
        self.assertLess(prepared[1].image.width, frames[1].width)
        self.assertFalse(prepared[1].blend, "clearing pixels must use exact no-blend replacement")
        self.assertTrue(prepared[2].blend, "opaque-only additions can inherit unchanged pixels")

    def test_identical_frames_are_coalesced_without_losing_time(self):
        first = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
        ImageDraw.Draw(first).rectangle((4, 4, 10, 10), fill=(255, 255, 255, 255))
        second = first.copy()
        third = first.copy()
        ImageDraw.Draw(third).point((15, 15), fill=(255, 0, 0, 255))
        prepared = _prepare_delta_frames([first, second, third], [30, 70, 100])
        self.assertEqual(len(prepared), 2)
        self.assertEqual(prepared[0].duration, 100)
        self.assertEqual(prepared[1].duration, 100)


if __name__ == "__main__":
    unittest.main()
