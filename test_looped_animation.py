from __future__ import annotations

import unittest

from PIL import Image

import advanced_pipeline as base
import app_all
import looped_animation
import operation_pipeline
import temporal_v2


class LoopedAnimationTests(unittest.TestCase):
    def records(self, count: int):
        return [
            base.FrameRecord(
                image=Image.new("RGBA", (8, 8), (index * 20, 10, 30, 255)),
                duration=100.0,
                source_indices={index},
            )
            for index in range(count)
        ]

    def test_token_is_removed_without_changing_manual_gap_text(self):
        enabled, cleaned = looped_animation.extract_looped_token(
            "1,looped-animation,3"
        )
        self.assertTrue(enabled)
        self.assertEqual(cleaned, "1,3")

    def test_conventional_processing_includes_ordinary_and_loop_gaps(self):
        records = self.records(4)
        plans, _, _ = temporal_v2._plans_for_job(
            records,
            {"target_gaps": "looped-animation", "smart_missing": False},
            4,
        )
        self.assertEqual([plan.kind for plan in plans].count("normal"), 3)
        self.assertEqual([plan.kind for plan in plans].count("loop"), 1)
        pairs = operation_pipeline.selected_pair_specs(records, plans, 4)
        self.assertEqual(len([pair for pair in pairs if not pair.wrap]), 3)
        self.assertEqual(len([pair for pair in pairs if pair.wrap]), 1)

    def test_manual_gap_plus_looped_animation_keeps_both(self):
        records = self.records(4)
        plans, _, _ = temporal_v2._plans_for_job(
            records,
            {"target_gaps": "2,looped-animation", "smart_missing": False},
            4,
        )
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0].kind, "normal")
        self.assertEqual(plans[0].source_gap, 1)
        self.assertEqual(plans[1].kind, "loop")

    def test_loop_is_not_suppressed_by_smart_missing_threshold(self):
        records = self.records(3)
        plans, _, _ = temporal_v2._plans_for_job(
            records,
            {
                "target_gaps": "looped-animation",
                "smart_missing": True,
                "missing_threshold": 1000000,
                "loop_analysis": "off",
                "alpha_threshold": 8,
            },
            3,
        )
        self.assertFalse(any(plan.kind == "normal" for plan in plans))
        self.assertEqual([plan.kind for plan in plans], ["loop"])

    def test_explicit_loop_toggle_is_present_in_served_workspace_script(self):
        self.assertIn(b"Looped animation", app_all.WORKSPACE_SCRIPT)
        self.assertIn(b"looped-animation-ui-v1", app_all.WORKSPACE_SCRIPT)


if __name__ == "__main__":
    unittest.main()
