from __future__ import annotations

import unittest
from types import SimpleNamespace

import repair_mode


class RepairModeTests(unittest.TestCase):
    def test_mode_tokens(self):
        self.assertEqual(repair_mode.repair_mode({}), "auto")
        self.assertEqual(repair_mode.repair_mode({"target_gaps": "1,repair-mode:manual"}), "manual")
        self.assertEqual(
            repair_mode.repair_mode({"target_gaps": "repair-mode:manual", "openai_repair_mode": "auto"}),
            "auto",
        )

    def test_manual_openai_pass_is_deferred_but_auto_runs(self):
        calls = []

        def original(records, stage_dir, *, settings, source_count, pass_number, pass_total, progress):
            calls.append(dict(settings))
            return {"engine": "openai", "repaired": 1}

        module = SimpleNamespace(
            _repair_pass=original,
            repair_engine=lambda settings: str(settings.get("repair_engine", "none")),
        )
        repair_mode.install(module)

        messages = []
        manual = module._repair_pass(
            [],
            ".",
            settings={"repair_engine": "openai", "target_gaps": "repair-mode:manual"},
            source_count=3,
            pass_number=2,
            pass_total=3,
            progress=lambda fraction, message: messages.append((fraction, message)),
        )
        self.assertTrue(manual["deferred"])
        self.assertEqual(manual["mode"], "manual")
        self.assertEqual(calls, [])
        self.assertIn("deferred", messages[-1][1].lower())

        automatic = module._repair_pass(
            [],
            ".",
            settings={"repair_engine": "openai", "target_gaps": "repair-mode:auto"},
            source_count=3,
            pass_number=2,
            pass_total=3,
            progress=lambda *_: None,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(automatic["mode"], "auto")


if __name__ == "__main__":
    unittest.main()
