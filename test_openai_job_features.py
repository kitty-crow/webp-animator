from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from openai_job_features import AuditorAwareBridgeClient, EnhancedOpenAIJobManager
from test_openai_interrogator import FakeBridge, png


class RecordingBridge(FakeBridge):
    def __init__(self, reject_first: bool = False):
        super().__init__(reject_first=reject_first)
        self.current_overrides = []
        self.audit_overrides_seen = []

    def set_auditor_overrides(self, overrides):
        self.current_overrides = list(overrides or [])

    def request(self, path, payload, **kwargs):
        if path == "/audit":
            self.audit_overrides_seen.append(list(self.current_overrides))
        return super().request(path, payload, **kwargs)


class OpenAIJobFeatureTests(unittest.TestCase):
    def make_manager(self, reject_first: bool = False):
        temporary = tempfile.TemporaryDirectory()
        manager = EnhancedOpenAIJobManager(Path(temporary.name))
        manager.bridge = RecordingBridge(reject_first=reject_first)
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
            self.assertEqual(public["auditor_overrides"][0]["type"], "test")
            self.assertEqual(public["auditor_overrides"][0]["description"], "wrong state")
        finally:
            temporary.cleanup()

    def test_override_is_sent_to_subsequent_audits(self):
        temporary, manager = self.make_manager(reject_first=True)
        try:
            created = manager.create(
                [("a.png", png((0, 0, 0, 255))), ("b.png", png((255, 255, 255, 255)))],
                {"mode": "fixed", "count": 2, "retry_policy": "manual", "auto_retries": 0, "max_spend_usd": 1},
            )
            rejected = self.wait(manager, created["id"])
            self.assertEqual(rejected["status"], "needs_review")
            self.assertEqual(manager.bridge.audit_overrides_seen[0], [])

            manager.accept_latest(created["id"])
            finished = self.wait(manager, created["id"])
            self.assertEqual(finished["status"], "done")
            self.assertGreaterEqual(len(manager.bridge.audit_overrides_seen), 2)
            learned = manager.bridge.audit_overrides_seen[-1]
            self.assertEqual(len(learned), 1)
            self.assertEqual(learned[0]["type"], "test")
            self.assertEqual(learned[0]["description"], "wrong state")
        finally:
            temporary.cleanup()

    def test_override_propagates_to_parent_batch_manifest(self):
        temporary, manager = self.make_manager(reject_first=True)
        try:
            created = manager.create(
                [("a.png", png((0, 0, 0, 255))), ("b.png", png((255, 255, 255, 255)))],
                {"mode": "single", "retry_policy": "manual", "auto_retries": 0, "max_spend_usd": 1},
            )
            rejected = self.wait(manager, created["id"])
            self.assertEqual(rejected["status"], "needs_review")

            batch_dir = Path(temporary.name) / ".openai-batches" / "a" * 32
            batch_dir.mkdir(parents=True)
            batch_manifest = batch_dir / "job.json"
            batch_manifest.write_text(json.dumps({
                "id": "a" * 32,
                "status": "needs_review",
                "request": {"scope": "all"},
                "current_child_id": created["id"],
                "tasks": [{"child_id": created["id"]}],
            }), encoding="utf-8")

            manager.accept_latest(created["id"])
            batch = json.loads(batch_manifest.read_text(encoding="utf-8"))
            learned = batch["request"]["auditor_overrides"]
            self.assertEqual(len(learned), 1)
            self.assertEqual(learned[0]["description"], "wrong state")
        finally:
            temporary.cleanup()

    def test_auditor_bridge_injects_learned_preferences_only_into_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            bridge = AuditorAwareBridgeClient(Path(temporary))
            bridge.set_auditor_overrides([{
                "type": "rendering_style",
                "region": "whole frame",
                "description": "anti-aliasing is acceptable because a pixel-art filter is applied later",
                "severity": 90,
            }])
            forwarded = []

            def fake_request(_self, path, payload, **_kwargs):
                forwarded.append((path, payload))
                return {"ok": True}

            with patch("openai_job_features.BridgeClient.request", new=fake_request):
                bridge.request("/plan", {"user_instruction": "original"})
                bridge.request("/audit", {"user_instruction": "original"})

            self.assertEqual(forwarded[0][1]["user_instruction"], "original")
            audit_instruction = forwarded[1][1]["user_instruction"]
            self.assertIn("AUDITOR JOB-LOCAL USER PREFERENCE OVERRIDE", audit_instruction)
            self.assertIn("anti-aliasing is acceptable", audit_instruction)
            self.assertIn("Continue to evaluate every unrelated", audit_instruction)

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
