from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import openai_batch
from openai_batch import OpenAIBatchManager, _atomic_json
from openai_interrogator import OpenAIJobManager
from test_openai_interrogator import FakeBridge, png


class OpenAIBatchTests(unittest.TestCase):
    def make_managers(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        children = OpenAIJobManager(root)
        children.bridge = FakeBridge()
        batches = OpenAIBatchManager(root, children)
        return temporary, children, batches

    def wait(self, batches: OpenAIBatchManager, job_id: str):
        deadline = time.time() + 8
        while time.time() < deadline:
            job = batches.get(job_id)
            if job and job.get("status") in {"done", "error", "needs_review", "budget_wait"}:
                return job
            time.sleep(0.02)
        self.fail("batch did not reach a stopped state")

    def test_range_runs_every_adjacent_gap(self):
        temporary, _children, batches = self.make_managers()
        try:
            uploads = [
                ("1.png", png((0, 0, 0, 255))),
                ("2.png", png((64, 64, 64, 255))),
                ("3.png", png((128, 128, 128, 255))),
                ("4.png", png((255, 255, 255, 255))),
            ]
            created = batches.create(
                uploads,
                {
                    "scope": "range",
                    "left_index": 0,
                    "right_index": 3,
                    "mode": "single",
                    "auto_retries": 0,
                    "max_spend_usd": 10,
                    "whole_sequence_context": True,
                },
            )
            job = self.wait(batches, created["id"])
            self.assertEqual(job["status"], "done")
            public = batches.public(job)
            self.assertEqual(len(public["segments"]), 3)
            self.assertEqual(len(public["accepted"]), 3)
            self.assertEqual(
                [(item["left_index"], item["right_index"]) for item in public["accepted"]],
                [(0, 1), (1, 2), (2, 3)],
            )
        finally:
            temporary.cleanup()

    def test_whole_loop_includes_closure_gap(self):
        temporary, _children, batches = self.make_managers()
        try:
            uploads = [
                ("1.png", png((0, 0, 0, 255))),
                ("2.png", png((128, 128, 128, 255))),
                ("3.png", png((255, 255, 255, 255))),
            ]
            created = batches.create(
                uploads,
                {
                    "scope": "all",
                    "sequence_mode": "loop",
                    "include_loop_closure": True,
                    "mode": "single",
                    "auto_retries": 0,
                    "max_spend_usd": 10,
                },
            )
            job = self.wait(batches, created["id"])
            self.assertEqual(job["status"], "done")
            public = batches.public(job)
            self.assertEqual(len(public["segments"]), 3)
            self.assertEqual(len(public["accepted"]), 3)
            closures = [item for item in public["accepted"] if item["loop_closure"]]
            self.assertEqual(len(closures), 1)
            self.assertEqual((closures[0]["left_index"], closures[0]["right_index"]), (2, 0))
        finally:
            temporary.cleanup()

    def test_segment_count_matches_scope(self):
        self.assertEqual(OpenAIBatchManager.segment_count({"scope": "range", "left_index": 1, "right_index": 4}, 6), 3)
        self.assertEqual(OpenAIBatchManager.segment_count({"scope": "all"}, 5), 4)
        self.assertEqual(
            OpenAIBatchManager.segment_count(
                {"scope": "all", "sequence_mode": "loop", "include_loop_closure": True},
                5,
            ),
            5,
        )

    def test_atomic_manifest_write_retries_windows_permission_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "job.json"
            original_replace = openai_batch.os.replace
            calls = 0

            def flaky_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError(13, "simulated Windows sharing violation", str(destination))
                return original_replace(source, destination)

            with patch("openai_batch.os.replace", side_effect=flaky_replace):
                _atomic_json(path, {"status": "running", "progress": 12})

            self.assertEqual(calls, 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["progress"], 12)
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
