from __future__ import annotations

import unittest

from PIL import Image

import advanced_pipeline as base
import operation_pipeline
import pipeline_settings
import temporal_v2


class OperationPipelineTests(unittest.TestCase):
    def records(self, count: int):
        return [
            base.FrameRecord(
                image=Image.new("RGBA", (4, 4), (index, index, index, 255)),
                duration=100.0,
                source_indices={index},
            )
            for index in range(count)
        ]

    def test_default_order_is_gap_repair_interpolation(self):
        settings = {
            "frame_generator": "speed",
            "interpolator": "rife",
            "target_gaps": "repair:propainter",
        }
        self.assertEqual(
            operation_pipeline.default_pipeline(settings),
            ["gap", "repair", "interpolate"],
        )

    def test_explicit_pipeline_preserves_repeated_reordered_passes(self):
        self.assertEqual(
            operation_pipeline.parse_pipeline("interpolate,gap,repair,gap,interpolate"),
            ["interpolate", "gap", "repair", "gap", "interpolate"],
        )

    def test_speed_two_frames_is_fixed_two_not_adaptive_zero(self):
        self.assertEqual(operation_pipeline._gap_count({"frames_to_fill": 2}), 2)
        self.assertEqual(operation_pipeline._gap_count({}), 1)
        self.assertEqual(operation_pipeline._gap_count({"frames_to_fill": 0}), 0)

    def test_five_source_frames_have_four_ordinary_gaps(self):
        records = self.records(5)
        plans, _, _ = temporal_v2._plans_for_job(records, {}, 5)
        pairs = operation_pipeline.selected_pair_specs(records, plans, 5)
        self.assertEqual(len(plans), 4)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(pairs) * operation_pipeline._gap_count({"frames_to_fill": 2}), 8)

    def test_five_frames_plus_loop_have_five_gap_pairs(self):
        records = self.records(5)
        ordinary, _, _ = temporal_v2._plans_for_job(records, {}, 5)
        plans = list(ordinary) + [temporal_v2.GapPlan("loop", None, 4, True)]
        pairs = operation_pipeline.selected_pair_specs(records, plans, 5)
        self.assertEqual(len(pairs), 5)
        self.assertEqual(len(pairs) * 2, 10)

    def test_pipeline_token_is_removed_from_gap_targets(self):
        pipeline, targets = pipeline_settings._extract("1,3,pipeline:gap>repair>interpolate,loop")
        self.assertEqual(pipeline, "gap,repair,interpolate")
        self.assertEqual(targets, "1,3,loop")


if __name__ == "__main__":
    unittest.main()
