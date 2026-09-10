from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image

import geometry_cache


class GeometryCacheTests(unittest.TestCase):
    def _png(self, path: Path, colour) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (5, 4), colour).save(path, format="PNG")

    def test_analysis_matching_is_reused_by_render_with_same_sources_and_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".webp-jobs" / ("a" * 32)
            analysis_paths = [
                root / "analysis" / "analysis-id" / "source" / "000000.png",
                root / "analysis" / "analysis-id" / "source" / "000001.png",
            ]
            render_paths = [
                root / "source" / "000000.png",
                root / "source" / "000001.png",
            ]
            self._png(analysis_paths[0], (10, 20, 30, 255))
            self._png(analysis_paths[1], (40, 50, 60, 255))
            self._png(render_paths[0], (10, 20, 30, 255))
            self._png(render_paths[1], (40, 50, 60, 255))

            calls = []
            advanced = types.SimpleNamespace()

            def raw_normalise(_legacy, images, _settings, progress=None):
                calls.append("match")
                if progress:
                    progress(40, "matched")
                return [image.copy() for image in images]

            advanced._normalise_geometry = raw_normalise
            temporal = types.SimpleNamespace()

            def analyse(_legacy, paths, settings, *args, **kwargs):
                images = [Image.open(path).convert("RGBA") for path in paths]
                return advanced._normalise_geometry(_legacy, images, settings, kwargs.get("progress"))

            def process(_legacy, _job_id, paths, settings):
                images = [Image.open(path).convert("RGBA") for path in paths]
                return advanced._normalise_geometry(_legacy, images, settings, None)

            temporal.analyse_paths = analyse
            temporal.process_job = process
            geometry_cache.install(advanced, temporal)

            settings = {
                "geometry_mode": "sequential",
                "axis": "xy",
                "max_shift": 64,
                "sigma": 24.0,
                "alpha_threshold": 8,
            }
            temporal.analyse_paths(None, analysis_paths, settings)
            temporal.process_job(None, "job", render_paths, settings)
            self.assertEqual(calls, ["match"])

    def test_changed_source_or_geometry_setting_does_not_reuse_stale_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".webp-jobs" / ("b" * 32)
            path = root / "source" / "000000.png"
            self._png(path, (10, 20, 30, 255))
            calls = []

            def normalise(_legacy, images, _settings, progress=None):
                calls.append("match")
                return [image.copy() for image in images]

            settings = {
                "geometry_mode": "sequential",
                "axis": "xy",
                "max_shift": 64,
                "sigma": 24.0,
                "alpha_threshold": 8,
            }
            cache_root = geometry_cache.cache_root_for_paths([path])
            images = [Image.open(path).convert("RGBA")]
            geometry_cache.normalise_with_cache(normalise, None, images, [path], settings, cache_root)
            geometry_cache.normalise_with_cache(normalise, None, images, [path], settings, cache_root)
            self.assertEqual(len(calls), 1)

            changed = dict(settings, max_shift=128)
            geometry_cache.normalise_with_cache(normalise, None, images, [path], changed, cache_root)
            self.assertEqual(len(calls), 2)

            self._png(path, (99, 20, 30, 255))
            changed_image = [Image.open(path).convert("RGBA")]
            geometry_cache.normalise_with_cache(normalise, None, changed_image, [path], changed, cache_root)
            self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
