from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from openai_job_features import AuditorAwareBridgeClient
from resilient_bridge import ResilientAuditorAwareBridgeClient


def png_b64(*, transparent: bool) -> str:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0) if transparent else (20, 30, 40, 255))
    if transparent:
        for y in range(8, 24):
            for x in range(10, 22):
                image.putpixel((x, y), (220, 160, 120, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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

    def test_transparent_black_hallucination_and_global_scale_are_not_rejections(self):
        transparent = png_b64(transparent=True)
        seen_payload = {}

        def fake_request(_self, path, payload, **_kwargs):
            seen_payload.update(payload or {})
            return {
                "audit": {
                    "acceptable": False,
                    "retry_recommended": True,
                    "observations": [],
                    "violations": [
                        {
                            "type": "background mismatch",
                            "region": "surround",
                            "description": "Red, yellow, white, dark, and transparent-looking graphic regions from the anchors are absent and replaced with black.",
                            "severity": 95,
                        },
                        {
                            "type": "framing",
                            "region": "whole frame",
                            "description": "The candidate appears modestly larger or differently positioned than the anchor framing, contrary to fixed camera, scale and placement.",
                            "severity": 80,
                        },
                    ],
                }
            }

        with tempfile.TemporaryDirectory() as temporary:
            bridge = ResilientAuditorAwareBridgeClient(Path(temporary))
            with patch.object(AuditorAwareBridgeClient, "request", new=fake_request):
                result = bridge.request("/audit", {
                    "auditor_effort": "medium",
                    "frame_a_b64": transparent,
                    "candidate_b64": transparent,
                    "frame_b_b64": transparent,
                    "user_instruction": "",
                    "context_frames": [],
                })

        audit = result["audit"]
        self.assertTrue(audit["acceptable"])
        self.assertTrue(audit["accepted_by_pipeline_tolerance"])
        self.assertEqual(audit["violations"], [])
        self.assertEqual(len(audit["pipeline_correctable_violations"]), 2)
        self.assertIn("HARD DECODED ALPHA FACTS", seen_payload["user_instruction"])
        self.assertIn("'has_transparency': True", seen_payload["user_instruction"])
        self.assertIn("optimises uniform resize and X/Y translation", seen_payload["user_instruction"])

    def test_real_structural_temporal_problem_is_not_hidden_by_pipeline_tolerance(self):
        transparent = png_b64(transparent=True)

        def fake_request(_self, _path, _payload, **_kwargs):
            return {
                "audit": {
                    "acceptable": False,
                    "retry_recommended": True,
                    "observations": [],
                    "violations": [
                        {
                            "type": "background",
                            "region": "canvas",
                            "description": "Transparent-looking background regions were replaced with black.",
                            "severity": 90,
                        },
                        {
                            "type": "temporal pose",
                            "region": "leg",
                            "description": "The leg reverses direction and creates a structurally impossible pose outside both anchor states.",
                            "severity": 92,
                        },
                    ],
                }
            }

        with tempfile.TemporaryDirectory() as temporary:
            bridge = ResilientAuditorAwareBridgeClient(Path(temporary))
            with patch.object(AuditorAwareBridgeClient, "request", new=fake_request):
                result = bridge.request("/audit", {
                    "auditor_effort": "medium",
                    "frame_a_b64": transparent,
                    "candidate_b64": transparent,
                    "frame_b_b64": transparent,
                    "context_frames": [],
                })

        audit = result["audit"]
        self.assertFalse(audit["acceptable"])
        self.assertEqual(len(audit["violations"]), 1)
        self.assertIn("structurally impossible", audit["violations"][0]["description"])

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
