from __future__ import annotations

import base64
import io
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from openai_interrogator import OpenAIJobManager, _balanced_targets


def png(colour: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGBA", (64, 64), colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeBridge:
    def __init__(self, reject_first: bool = False):
        self.reject_first = reject_first
        self.audits = 0
        self.plans = 0

    def status(self):
        return {"key_configured": True, "schema_built": True, "runtime": "fake", "bridge_running": True}

    def request(self, path, payload, **_kwargs):
        if path == "/estimate":
            attempts = int(payload.get("attempts", 1))
            return {
                "one_attempt": 0.05,
                "maximum": 0.05 * attempts,
                "attempts": attempts,
                "planner": 0.001,
                "generation": 0.048,
                "auditor": 0.001,
            }
        if path == "/plan":
            self.plans += 1
            return {
                "plan": {
                    "valid_pair": True,
                    "sequence_summary": "generic test sequence",
                    "transition_summary": "state changes between anchors",
                    "refined_user_intent": str(payload.get("user_instruction", "")),
                    "target_fraction": float(payload.get("target_fraction", 0.5)),
                    "invariants": ["canvas"],
                    "changing_regions": [],
                    "generation_instruction": "create the requested temporal midpoint",
                    "risks": [],
                    "additional_frame_worthwhile": self.plans < 3,
                    "benefit_score": 80 if self.plans < 3 else 10,
                    "recommended_next_fraction": 0.5,
                    "stop_reason": "",
                },
                "usage": {"inputTokens": 10, "outputTokens": 10},
                "model": "fake",
            }
        if path == "/generate":
            return {
                "image_b64": base64.b64encode(png((128, 128, 128, 255))).decode("ascii"),
                "mime": "image/png",
                "usage": {"input_tokens": 10, "output_tokens": 10},
                "model": "fake-image",
            }
        if path == "/audit":
            self.audits += 1
            accepted = not (self.reject_first and self.audits == 1)
            return {
                "audit": {
                    "acceptable": accepted,
                    "overall_score": 95 if accepted else 40,
                    "temporal_position_score": 95,
                    "source_consistency_score": 95,
                    "instruction_match_score": 95 if accepted else 30,
                    "invariant_preservation_score": 95,
                    "structural_coherence_score": 95,
                    "unnecessary_change_score": 95,
                    "observations": [],
                    "violations": [] if accepted else [{"type": "test", "region": "test", "description": "wrong state", "severity": 80}],
                    "keep_from_attempt": ["canvas"],
                    "change_on_retry": [] if accepted else ["temporal state"],
                    "retry_recommended": not accepted,
                    "additional_frame_worthwhile": True,
                    "benefit_score": 80,
                    "corrective_instruction": "correct the temporal state" if not accepted else "",
                },
                "usage": {"inputTokens": 10, "outputTokens": 10},
                "model": "fake",
            }
        if path == "/cost":
            return {"planner": 0.001, "generation": 0.048, "auditor": 0.001}
        raise AssertionError(path)


class OpenAIInterrogatorTests(unittest.TestCase):
    def make_manager(self, reject_first: bool = False):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        manager = OpenAIJobManager(root)
        manager.bridge = FakeBridge(reject_first=reject_first)
        return temporary, manager

    def wait(self, manager: OpenAIJobManager, job_id: str, terminal: set[str] | None = None):
        terminal = terminal or {"done", "error", "needs_review", "budget_wait"}
        deadline = time.time() + 5
        while time.time() < deadline:
            job = manager.get(job_id)
            if job and job.get("status") in terminal:
                return job
            time.sleep(0.02)
        self.fail("job did not reach a terminal/review state")

    def test_balanced_fixed_targets(self):
        self.assertEqual(_balanced_targets(1), [0.5])
        self.assertEqual(_balanced_targets(3), [0.5, 0.25, 0.75])

    def test_fixed_job_persists_and_accepts_all_targets(self):
        temporary, manager = self.make_manager()
        try:
            created = manager.create(
                [("a.png", png((0, 0, 0, 255))), ("b.png", png((255, 255, 255, 255)))],
                {"mode": "fixed", "count": 3, "auto_retries": 0, "max_spend_usd": 1},
            )
            job = self.wait(manager, created["id"])
            self.assertEqual(job["status"], "done")
            self.assertEqual(
                sorted(round(float(item["target_fraction"]), 2) for item in job["accepted"]),
                [0.25, 0.5, 0.75],
            )
            self.assertTrue((Path(temporary.name) / ".openai-jobs" / created["id"] / "job.json").is_file())
        finally:
            temporary.cleanup()

    def test_rejected_attempt_waits_for_feedback_then_reuses_history(self):
        temporary, manager = self.make_manager(reject_first=True)
        try:
            created = manager.create(
                [("a.png", png((0, 0, 0, 255))), ("b.png", png((255, 255, 255, 255)))],
                {"mode": "single", "auto_retries": 0, "max_spend_usd": 1},
            )
            first = self.wait(manager, created["id"])
            self.assertEqual(first["status"], "needs_review")
            self.assertEqual(len(first["attempts"]), 1)
            self.assertFalse(first["attempts"][0]["accepted"])

            manager.resume(created["id"], "The previous temporal state was wrong.", 1)
            second = self.wait(manager, created["id"])
            self.assertEqual(second["status"], "done")
            self.assertEqual(len(second["attempts"]), 2)
            self.assertTrue(second["attempts"][1]["accepted"])
            self.assertEqual(len(second["accepted"]), 1)
        finally:
            temporary.cleanup()

    def test_budget_gate_stops_before_generation(self):
        temporary, manager = self.make_manager()
        try:
            created = manager.create(
                [("a.png", png((0, 0, 0, 255))), ("b.png", png((255, 255, 255, 255)))],
                {"mode": "single", "auto_retries": 0, "max_spend_usd": 0.01},
            )
            job = self.wait(manager, created["id"])
            self.assertEqual(job["status"], "budget_wait")
            self.assertEqual(len(job["attempts"]), 0)
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
