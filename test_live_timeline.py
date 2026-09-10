from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

import live_timeline


class LiveTimelineTests(unittest.TestCase):
    def _png(self, path: Path, colour):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (4, 4), colour).save(path)

    def test_generated_frame_is_inserted_between_originals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pass_dir = root / "pass-01-gap"
            first = pass_dir / "input" / "000000.png"
            second = pass_dir / "input" / "000001.png"
            generated = pass_dir / "speed-generator-v2" / "frames" / "0000_0000.png"
            self._png(first, (255, 0, 0, 255))
            self._png(second, (0, 0, 255, 255))
            self._png(generated, (128, 0, 128, 255))

            base = {
                "pass_number": 1,
                "operation": "gap",
                "engine": "speed",
                "frames": [
                    {"key": "source:0", "name": "one.png", "rel": first.relative_to(root).as_posix(), "stage": "Original"},
                    {"key": "source:1", "name": "two.png", "rel": second.relative_to(root).as_posix(), "stage": "Original"},
                ],
                "pairs": [
                    {"task_index": 0, "left_index": 0, "right_index": 1, "wrap": False, "source_gap": 0},
                ],
            }
            timeline = live_timeline._dynamic_engine(root, "abc", pass_dir, base)
            self.assertEqual([item["key"] for item in timeline], ["source:0", "pass:1:task:0:frame:0", "source:1"])
            self.assertEqual(timeline[1]["stage"], "Generated · SPEED")

    def test_propainter_replaces_existing_timeline_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pass_dir = root / "pass-02-repair"
            original = pass_dir / "input" / "000000.png"
            repaired = pass_dir / "temporal-repair" / "output" / "frames" / "000000.png"
            self._png(original, (10, 20, 30, 255))
            self._png(repaired, (40, 50, 60, 255))

            base = {
                "pass_number": 2,
                "operation": "repair",
                "engine": "propainter",
                "frames": [
                    {
                        "key": "pass:1:generated:0",
                        "name": "middle.png",
                        "rel": original.relative_to(root).as_posix(),
                        "stage": "Generated · SPEED",
                        "generated": True,
                    },
                ],
                "pairs": [],
            }
            timeline = live_timeline._dynamic_repair(root, "abc", pass_dir, base)
            self.assertEqual(len(timeline), 1)
            self.assertEqual(timeline[0]["key"], "pass:1:generated:0")
            self.assertIn("ProPainter", timeline[0]["stage"])
            self.assertIn("Generated · SPEED", timeline[0]["stage"])
            self.assertIn("temporal-repair/output/frames/000000.png", timeline[0]["url"])


if __name__ == "__main__":
    unittest.main()
