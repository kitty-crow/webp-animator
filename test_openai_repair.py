from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import openai_repair


class OpenAIRepairTests(unittest.TestCase):
    def test_api_mask_makes_only_selected_pixels_editable(self):
        selection = Image.new("L", (3, 2), 0)
        selection.putpixel((1, 0), 255)
        api_mask = openai_repair._api_edit_mask(selection)
        alpha = api_mask.getchannel("A")
        self.assertEqual(alpha.getpixel((1, 0)), 0)
        self.assertEqual(alpha.getpixel((0, 0)), 255)
        self.assertEqual(alpha.getpixel((2, 1)), 255)

    def test_constrained_result_preserves_candidate_alpha_exactly(self):
        candidate = Image.new("RGBA", (3, 2), (10, 20, 30, 255))
        candidate.putpixel((1, 0), (10, 20, 30, 64))
        candidate.putpixel((2, 1), (10, 20, 30, 0))
        repaired = Image.new("RGBA", candidate.size, (200, 210, 220, 255))

        result = openai_repair._constrain_result(candidate, repaired, None)

        self.assertEqual(list(result.getchannel("A").getdata()), list(candidate.getchannel("A").getdata()))
        self.assertEqual(result.getpixel((0, 0))[:3], (200, 210, 220))
        self.assertEqual(result.getpixel((1, 0))[3], 64)
        self.assertEqual(result.getpixel((2, 1))[3], 0)

    def test_manual_selection_is_hard_local_constraint_after_generation(self):
        candidate = Image.new("RGBA", (4, 3), (10, 20, 30, 255))
        candidate.putpixel((2, 1), (10, 20, 30, 77))
        repaired = Image.new("RGBA", candidate.size, (230, 80, 40, 255))
        selection = Image.new("L", candidate.size, 0)
        selection.putpixel((2, 1), 255)

        result = openai_repair._constrain_result(candidate, repaired, selection)

        for y in range(candidate.height):
            for x in range(candidate.width):
                if (x, y) == (2, 1):
                    self.assertEqual(result.getpixel((x, y)), (230, 80, 40, 77))
                else:
                    self.assertEqual(result.getpixel((x, y)), candidate.getpixel((x, y)))

    def test_browser_rgba_mask_normalises_any_painted_alpha_to_selection(self):
        mask = Image.new("RGBA", (3, 1), (255, 48, 80, 0))
        mask.putpixel((1, 0), (255, 48, 80, 64))
        mask.putpixel((2, 0), (255, 48, 80, 220))
        selection = openai_repair._selection_mask(mask, mask.size)
        self.assertEqual(list(selection.getdata()), [0, 255, 255])

    def test_api_key_is_read_from_app_dotenv_without_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            dotenv = Path(temporary) / ".env"
            dotenv.write_text("OPENAI_API_KEY=sk-test-first\n", encoding="utf-8")
            with mock.patch.object(openai_repair, "DOTENV_PATH", dotenv), mock.patch.dict(
                os.environ, {"OPENAI_API_KEY": ""}, clear=False
            ):
                self.assertEqual(openai_repair._api_key(), "sk-test-first")
                self.assertTrue(openai_repair.status()["ready"])
                dotenv.write_text("OPENAI_API_KEY=sk-test-second\n", encoding="utf-8")
                self.assertEqual(openai_repair._api_key(), "sk-test-second")

    def test_nonempty_process_environment_overrides_dotenv(self):
        with tempfile.TemporaryDirectory() as temporary:
            dotenv = Path(temporary) / ".env"
            dotenv.write_text("OPENAI_API_KEY=sk-file\n", encoding="utf-8")
            with mock.patch.object(openai_repair, "DOTENV_PATH", dotenv), mock.patch.dict(
                os.environ, {"OPENAI_API_KEY": "sk-process"}, clear=False
            ):
                self.assertEqual(openai_repair._api_key(), "sk-process")


if __name__ == "__main__":
    unittest.main()
