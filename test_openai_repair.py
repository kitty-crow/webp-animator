from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import openai_repair
import openai_repair_context


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

    def test_api_canvas_rounds_each_dimension_up_to_multiple_of_16(self):
        self.assertEqual(openai_repair_context._multiple_of_16(1193), 1200)
        self.assertEqual(openai_repair_context._multiple_of_16(1663), 1664)
        self.assertEqual(openai_repair_context._multiple_of_16(1024), 1024)

    def test_context_sheet_contains_every_animation_frame_and_is_api_aligned(self):
        frames = [Image.new("RGBA", (1193, 1663), (index * 20, 40, 80, 255)) for index in range(10)]
        sheet = openai_repair_context._context_sheet(frames, 5)
        self.assertEqual(sheet.width % 16, 0)
        self.assertEqual(sheet.height % 16, 0)
        self.assertGreater(sheet.width, 0)
        self.assertGreater(sheet.height, 0)

    def test_auto_guard_rejects_a_broad_repaint_and_keeps_candidate(self):
        candidate = Image.new("RGBA", (20, 20), (10, 20, 30, 255))
        hallucinated = Image.new("RGBA", (20, 20), (240, 210, 180, 255))
        with mock.patch.dict(os.environ, {"OPENAI_REPAIR_MAX_AUTO_CHANGE": "0.18"}):
            result, guard = openai_repair_context._constrain_and_guard(candidate, hallucinated, None)
        self.assertFalse(guard["accepted"])
        self.assertEqual(guard["guard"], "broad-repaint-rejected")
        self.assertEqual(list(result.getdata()), list(candidate.getdata()))

    def test_manual_mask_bypasses_broad_repaint_but_only_inside_mask(self):
        candidate = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
        hallucinated = Image.new("RGBA", (8, 8), (220, 40, 60, 255))
        selection = Image.new("L", candidate.size, 0)
        selection.putpixel((3, 4), 255)
        result, guard = openai_repair_context._constrain_and_guard(candidate, hallucinated, selection)
        self.assertTrue(guard["accepted"])
        self.assertEqual(result.getpixel((3, 4))[:3], (220, 40, 60))
        self.assertEqual(result.getpixel((0, 0)), candidate.getpixel((0, 0)))

    def test_repair_prompt_forbids_ghosting_and_reanimation(self):
        prompt = openai_repair_context.REPAIR_PROMPT.lower()
        self.assertIn("do not add motion blur", prompt)
        self.assertIn("ghost images", prompt)
        self.assertIn("not allowed to invent a new in-between frame", prompt)

    def test_dotenv_key_is_visible_without_process_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            dotenv = Path(temporary) / ".env"
            with mock.patch.object(openai_repair, "DOTENV_PATH", dotenv):
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("OPENAI_API_KEY", None)
                    dotenv.write_text("OPENAI_API_KEY=first-key\n", encoding="utf-8")
                    self.assertEqual(openai_repair._api_key(), "first-key")
                    dotenv.write_text("OPENAI_API_KEY=second-key\n", encoding="utf-8")
                    self.assertEqual(openai_repair._api_key(), "second-key")


if __name__ == "__main__":
    unittest.main()
