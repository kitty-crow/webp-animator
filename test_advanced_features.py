from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops

import advanced_pipeline
import app
import app_all
from global_jobs import GlobalJobStore


class AdvancedFeatureTests(unittest.TestCase):
    def test_new_settings_parse_independently(self):
        settings = app_all._settings_from_fields({
            "geometry_mode": "none",
            "interpolator": "amt",
            "frame_generator": "speed",
            "rife_multiplier": "4",
            "smart_reduction": "on",
            "reduction_threshold": "3.5",
            "smart_missing": "on",
            "missing_threshold": "14.5",
            "target_gaps": "1,4,loop",
            "frames_to_fill": "0",
            "loop_analysis": "alongside",
        })
        self.assertEqual(settings["geometry_mode"], "none")
        self.assertEqual(settings["interpolator"], "amt")
        self.assertEqual(settings["frame_generator"], "speed")
        self.assertEqual(settings["rife_multiplier"], 4)
        self.assertTrue(settings["smart_reduction"])
        self.assertTrue(settings["smart_missing"])
        self.assertEqual(settings["target_gaps"], "1,4,loop")
        self.assertEqual(settings["frames_to_fill"], 0)
        self.assertEqual(settings["loop_analysis"], "alongside")

    def test_old_persisted_rife_setting_remains_compatible(self):
        settings = app_all._settings_from_fields({"rife_multiplier": "4"})
        self.assertEqual(settings["interpolator"], "rife")
        self.assertEqual(settings["rife_multiplier"], 4)
        self.assertNotIn("frames_to_fill", settings)
        self.assertEqual(settings["loop_analysis"], "off")

    def test_analysis_staging_is_isolated_from_render_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_global = app_all.GLOBAL
            try:
                app_all.GLOBAL = GlobalJobStore(Path(temporary))
                job_id = app_all.GLOBAL.new_id()
                _, render_paths = app_all.GLOBAL.save_sources(
                    job_id,
                    [("render.png", b"render-source")],
                    {"duration": 100},
                )
                render_path = render_paths[0]
                analysis_id = "a" * 32
                analysis_paths, sources = app_all._stage_analysis_sources(
                    job_id,
                    analysis_id,
                    [("analysis.png", b"analysis-source")],
                )

                self.assertEqual(render_path.read_bytes(), b"render-source")
                self.assertEqual(analysis_paths[0].read_bytes(), b"analysis-source")
                self.assertNotEqual(render_path.parent, analysis_paths[0].parent)
                self.assertIn("analysis", sources[0]["path"])
            finally:
                app_all.GLOBAL = old_global

    def test_target_gap_parser_accepts_display_pairs(self):
        self.assertEqual(advanced_pipeline.parse_target_gaps("1-2, 4-5"), {0, 3})

    def test_geometry_none_does_not_move_or_resize_source_content(self):
        first = Image.new("RGBA", (10, 8), (0, 0, 0, 0))
        first.putpixel((2, 3), (255, 0, 0, 255))
        second = Image.new("RGBA", (20, 12), (0, 0, 0, 0))
        second.putpixel((2, 3), (0, 255, 0, 255))
        output = advanced_pipeline._normalise_geometry(
            app,
            [first, second],
            {
                "geometry_mode": "none",
                "axis": "xy",
                "max_shift": 64,
                "sigma": 24.0,
                "alpha_threshold": 8,
            },
            None,
        )
        self.assertEqual([frame.size for frame in output], [(20, 12), (20, 12)])
        self.assertEqual(output[0].getpixel((2, 3)), (255, 0, 0, 255))
        self.assertEqual(output[1].getpixel((2, 3)), (0, 255, 0, 255))

    def test_selective_recursive_frames_keep_true_timing(self):
        left = Image.new("RGBA", (8, 8), (0, 0, 0, 255))
        right = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
        records = [
            advanced_pipeline.FrameRecord(left, 100.0, {0}),
            advanced_pipeline.FrameRecord(right, 100.0, {1}),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            first_mid = Path(temporary) / "mid-50.png"
            second_mid = Path(temporary) / "mid-75.png"
            Image.new("RGBA", (8, 8), (128, 128, 128, 255)).save(first_mid)
            Image.new("RGBA", (8, 8), (192, 192, 192, 255)).save(second_mid)
            advanced_pipeline._insert_interpolator_results(
                records,
                [("task", 0, 0, 2)],
                {
                    "task": [
                        {"path": str(first_mid), "t": 0.5},
                        {"path": str(second_mid), "t": 0.75},
                    ]
                },
                "rife",
            )
        self.assertEqual(len(records), 4)
        self.assertAlmostEqual(records[0].duration, 50.0)
        self.assertAlmostEqual(records[1].duration, 25.0)
        self.assertAlmostEqual(records[2].duration, 25.0)
        self.assertAlmostEqual(records[3].duration, 100.0)


if __name__ == "__main__":
    unittest.main()
