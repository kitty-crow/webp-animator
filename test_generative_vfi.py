from __future__ import annotations

import os
import unittest

import generative_vfi
import generative_models_ui
import tooncrafter_vfi


tooncrafter_vfi.install_backend()


class FakeLegacy:
    def __init__(self, fields):
        self._fields = dict(fields)

    def parse_multipart(self, _content_type, _body):
        return dict(self._fields), [("frames", "one.png", b"x")]


class GenerativeVFITests(unittest.TestCase):
    def test_marker_round_trip(self):
        value = generative_vfi._append_marker("1,loop", "resshift")
        self.assertIn("__vfi_resshift__", value)
        self.assertEqual(generative_vfi._marker_engine(value), "resshift")
        self.assertEqual(generative_vfi._marker_engine("1,2,loop"), None)

        toon = generative_vfi._append_marker("2", "tooncrafter")
        self.assertIn("__vfi_tooncrafter__", toon)
        self.assertEqual(generative_vfi._marker_engine(toon), "tooncrafter")

    def test_parse_bridge_preserves_gap_selection(self):
        legacy = FakeLegacy({"interpolator": "mog_ani", "target_gaps": "2,loop"})
        generative_vfi._install_parse_bridge(legacy)
        fields, files = legacy.parse_multipart("multipart/form-data", b"")
        self.assertEqual(fields["interpolator"], "amt")
        self.assertIn("__vfi_mog_ani__", fields["target_gaps"])
        self.assertIn("2", fields["target_gaps"])
        self.assertIn("loop", fields["target_gaps"])
        self.assertEqual(len(files), 1)

    def test_tooncrafter_parse_bridge_uses_same_gap_context(self):
        legacy = FakeLegacy({"interpolator": "tooncrafter", "target_gaps": "1,4"})
        generative_vfi._install_parse_bridge(legacy)
        fields, _ = legacy.parse_multipart("multipart/form-data", b"")
        self.assertEqual(fields["interpolator"], "amt")
        self.assertIn("__vfi_tooncrafter__", fields["target_gaps"])
        self.assertIn("1", fields["target_gaps"])
        self.assertIn("4", fields["target_gaps"])

    def test_non_generative_interpolator_is_untouched(self):
        legacy = FakeLegacy({"interpolator": "rife", "target_gaps": "3"})
        generative_vfi._install_parse_bridge(legacy)
        fields, _ = legacy.parse_multipart("multipart/form-data", b"")
        self.assertEqual(fields["interpolator"], "rife")
        self.assertEqual(fields["target_gaps"], "3")

    def test_ui_patch_contains_all_advanced_models(self):
        for value in ("resshift", "mog_ani", "mog_real", "tooncrafter"):
            self.assertIn(value, generative_models_ui.UI_PATCH)

    def test_visible_device_tokens_honour_parent_mask(self):
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "3,1"
            self.assertEqual(generative_vfi._visible_gpu_tokens(), ["3", "1"])
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


if __name__ == "__main__":
    unittest.main()
