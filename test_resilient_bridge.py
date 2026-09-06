from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openai_job_features import AuditorAwareBridgeClient
from resilient_bridge import ResilientAuditorAwareBridgeClient


class ResilientBridgeTests(unittest.TestCase):
    def test_planner_retries_with_more_tokens_then_trims_context(self):
        calls = []

        def fake_request(_self, path, payload, **_kwargs):
            calls.append((path, dict(payload or {})))
            if len(calls) < 3:
                raise RuntimeError(
                    "OpenAI bridge request failed (500): Could not parse the expected output after 2 attempts: OpenAI returned no output text"
                )
            return {"plan": {"valid_pair": True}}

        with tempfile.TemporaryDirectory() as temporary:
            bridge = ResilientAuditorAwareBridgeClient(Path(temporary))
            with patch.object(AuditorAwareBridgeClient, "request", new=fake_request):
                result = bridge.request("/plan", {
                    "planner_effort": "medium",
                    "context_frames": [{"label": "timeline"}],
                })

        self.assertTrue(result["plan"]["valid_pair"])
        self.assertEqual([call[1]["max_output_tokens"] for call in calls], [6000, 10000, 12000])
        self.assertEqual([call[1]["planner_effort"] for call in calls], ["medium", "low", "low"])
        self.assertEqual(calls[0][1]["context_frames"], [{"label": "timeline"}])
        self.assertEqual(calls[1][1]["context_frames"], [{"label": "timeline"}])
        self.assertEqual(calls[2][1]["context_frames"], [])
        self.assertEqual(result["adaptive_structured_output_recovery"]["attempt"], 3)

    def test_auditor_gets_same_adaptive_recovery(self):
        calls = []

        def fake_request(_self, path, payload, **_kwargs):
            calls.append((path, dict(payload or {})))
            if len(calls) == 1:
                raise RuntimeError(
                    "OpenAI bridge request failed (500): Could not parse the expected output after 2 attempts: OpenAI returned no output text"
                )
            return {"audit": {"acceptable": True}}

        with tempfile.TemporaryDirectory() as temporary:
            bridge = ResilientAuditorAwareBridgeClient(Path(temporary))
            with patch.object(AuditorAwareBridgeClient, "request", new=fake_request):
                result = bridge.request("/audit", {
                    "auditor_effort": "medium",
                    "context_frames": [{"label": "timeline"}],
                })

        self.assertTrue(result["audit"]["acceptable"])
        self.assertEqual([call[1]["max_output_tokens"] for call in calls], [5000, 8000])
        self.assertEqual([call[1]["auditor_effort"] for call in calls], ["medium", "low"])
        self.assertEqual(result["adaptive_structured_output_recovery"]["attempt"], 2)

    def test_unrelated_errors_are_not_retried(self):
        calls = []

        def fake_request(_self, path, payload, **_kwargs):
            calls.append((path, payload))
            raise RuntimeError("OpenAI bridge request failed (401): invalid API key")

        with tempfile.TemporaryDirectory() as temporary:
            bridge = ResilientAuditorAwareBridgeClient(Path(temporary))
            with patch.object(AuditorAwareBridgeClient, "request", new=fake_request):
                with self.assertRaisesRegex(RuntimeError, "invalid API key"):
                    bridge.request("/audit", {"auditor_effort": "medium"})

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
