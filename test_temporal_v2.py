from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

import advanced_pipeline as base
import temporal_v2
from worker_common import adaptive_midpoint_fill, fixed_midpoint_fill


class TemporalV2Tests(unittest.TestCase):
    def test_parse_target_gap_spec_accepts_loop(self):
        normal, loop = temporal_v2.parse_target_gap_spec("1,3,loop")
        self.assertEqual(normal, {0, 2})
        self.assertTrue(loop)

    def test_legacy_fill_count_stays_multiplier_based_without_gap_fill(self):
        self.assertEqual(temporal_v2._requested_fill_count({"rife_multiplier": 4}, False), 3)
        self.assertEqual(
            temporal_v2._requested_fill_count({"rife_multiplier": 8, "frames_to_fill": 1}, True),
            1,
        )

    def test_fixed_midpoint_fill_honours_arbitrary_positive_count(self):
        first = Image.new("RGBA", (8, 8), (0, 0, 0, 255))
        second = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
        saved = []

        def generate(a, b):
            return Image.blend(a, b, 0.5)

        def save(image, t):
            ref = {"image": image, "t": t}
            saved.append(ref)
            return ref

        refs = fixed_midpoint_fill(
            first,
            second,
            count=5,
            generate_midpoint=generate,
            save_midpoint=save,
        )
        self.assertEqual(len(refs), 5)
        self.assertEqual(sorted(ref["t"] for ref in refs), [ref["t"] for ref in refs])

    def test_adaptive_fill_can_decide_zero_frames(self):
        first = Image.new("RGBA", (8, 8), (10, 10, 10, 255))
        second = first.copy()

        outcome = adaptive_midpoint_fill(
            first,
            second,
            threshold=1.0,
            max_frames=31,
            generate_midpoint=lambda a, b: Image.blend(a, b, 0.5),
            save_midpoint=lambda image, t: {"t": t},
        )
        self.assertEqual(outcome["frames"], [])
        self.assertTrue(outcome["satisfied"])

    def test_loop_insert_subdivides_existing_last_interval(self):
        first = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
        last = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
        records = [
            base.FrameRecord(first, 100.0, source_indices={0}),
            base.FrameRecord(last, 100.0, source_indices={1}),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mid.png"
            Image.new("RGBA", (4, 4), (128, 128, 128, 255)).save(path)
            plan = temporal_v2.GapPlan("loop", None, 1, True)
            temporal_v2._insert_gap_frames(
                records,
                plan,
                [{"path": str(path), "t": 0.5}],
                engine_label="rife",
                source_count=2,
            )

        self.assertEqual(len(records), 3)
        self.assertAlmostEqual(records[1].duration, 50.0)
        self.assertAlmostEqual(records[2].duration, 50.0)
        self.assertEqual(records[2].between_source_frames, (1, 0))
        self.assertAlmostEqual(sum(record.duration for record in records), 200.0)


if __name__ == "__main__":
    unittest.main()
