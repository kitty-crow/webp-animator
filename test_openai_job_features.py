from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from openai_job_features import EnhancedOpenAIJobManager
from test_openai_interrogator import FakeBridge, png


class OpenAIJobFeatureTests(unittest.TestCase):
    def make_manager(self, reject_first: bool = False):
        temporary = tempfile.TemporaryDirectory()
        manager = EnhancedOpenAIJobManager(Path(temporary.name))
        manager.bridge = FakeBridge(reject_first=reject_first)
        return temporary, manager

    def wait(self, manager: EnhancedOpenAIJobManager, job_id: str, terminal=None):
        terminal = terminal or {"done", "error", "needs_review", "budget_wait"}
        deadline = time.time() + 8
        while time.time() < deadline:
            job = manager.get(job_id)
            if job and job.get("status") in terminal:
                return job
            time.sleep(0.02)
        self.fail("job did not reach a stopped state")

    def test_auto_budget_retries_rejection_without_user_intervention(self):
        temporary, manager = self.make_manager(reject_first=True)
        try:
            created = manager.create(
                [("a.png", png((0, 0, 0, 255))), ("b.png", png((255, 255, 255, 255)))],
                {
                    "mode": "single",
                    "retry_policy": "auto_budget",
                    "auto_retries": 0,
                    "max_spend_usd": 1,
                },
            )
            job = self.wait(manager, created["id"])
            self.assertEqual(job["status"], "done")
            self.assertEqual(len(job["attempts"]), 2)
            self.assertFalse(job["attempts"][0]["accepted"])
            self.assertTrue(job["attempts"][1]["accepted"])
        finally:
            temporary.cleanup()

    def test_auto_budget_requires_finite_positive_budget(self):
        temporary, manager = self.make_manager()
        try:
            with self.assertRaisesRegex(ValueError, "maximum OpenAI spend"):
                manager.create(
                    [("a.png", png((0, 0, 0, 255))), ("b.png", png((255, 255, 255, 255)))],
                    {"mode": "single", "retry_policy": "auto_budget", "max_spend_usd": 0},
                )
        finally:
            temporary.cleanup()

    def test_user_can_accept_rejected_candidate_and_continue(self):
        temporary, manager = self.make_manager(reject_first=True)
        try:
            created = manager.create(
                [("a.png", png((0, 0, 0, 255))), ("b.png", png((255, 255, 255, 255)))],
                {"mode": "single", "retry_policy": "manual", "auto_retries": 0, "max_spend_usd": 1},
            )
            rejected = self.wait(manager, created["id"])
            self.assertEqual(rejected["status"], "needs_review")
            self.assertFalse(rejected["attempts"][-1]["accepted"])

            manager.accept_latest(created["id"])
            finished = self.wait(manager, created["id"])
            self.assertEqual(finished["status"], "done")
            self.assertEqual(len(finished["accepted"]), 1)
            self.assertTrue(finished["attempts"][-1]["accepted_by_user"])
            public = manager.public(finished)
            self.assertTrue(public["attempts"][-1]["accepted_by_user"])
            self.assertTrue(public["accepted"][0]["accepted_by_user"])
        finally:
            temporary.cleanup()

    def test_public_job_exposes_recoverable_source_frames(self):
        temporary, manager = self.make_manager()
        try:
            created = manager.create(
                [("one.png", png((0, 0, 0, 255))), ("two.png", png((255, 255, 255, 255)))],
                {"mode": "single", "auto_retries": 0, "max_spend_usd": 1},
            )
            job = manager.get(created["id"])
            self.assertIsNotNone(job)
            public = manager.public(job)
            self.assertEqual(public["source_count"], 2)
            self.assertEqual(public["source_names"], ["one.png", "two.png"])
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
