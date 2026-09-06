from __future__ import annotations

from copy import deepcopy
from typing import Any

from openai_job_features import AuditorAwareBridgeClient


_STRUCTURED_OUTPUT_MARKERS = (
    "Could not parse the expected output",
    "OpenAI returned no output text",
    "Structured output was unavailable",
)


class ResilientAuditorAwareBridgeClient(AuditorAwareBridgeClient):
    """Retry planner/auditor structured-output failures with adaptive budgets.

    The vendored OpenAISchema wrapper currently retries the same request internally
    when structured output is missing. This outer recovery layer changes the request
    on subsequent attempts instead of repeating the same failure: larger output budget,
    lower reasoning effort, and finally a reduced timeline context while retaining the
    immediate anchors (and candidate for audits).
    """

    @staticmethod
    def _is_structured_output_failure(error: BaseException) -> bool:
        message = str(error)
        return any(marker in message for marker in _STRUCTURED_OUTPUT_MARKERS)

    @staticmethod
    def _variants(path: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if path == "/plan":
            effort_key = "planner_effort"
            requested_effort = str(payload.get(effort_key, "medium"))
            budgets = (
                (requested_effort, max(6000, int(payload.get("max_output_tokens", 0) or 0)), False),
                ("low", 10000, False),
                ("low", 12000, True),
            )
        elif path == "/audit":
            effort_key = "auditor_effort"
            requested_effort = str(payload.get(effort_key, "medium"))
            budgets = (
                (requested_effort, max(5000, int(payload.get("max_output_tokens", 0) or 0)), False),
                ("low", 8000, False),
                ("low", 10000, True),
            )
        else:
            return [deepcopy(payload)]

        variants: list[dict[str, Any]] = []
        for effort, max_tokens, trim_context in budgets:
            candidate = deepcopy(payload)
            candidate[effort_key] = effort
            candidate["max_output_tokens"] = max_tokens
            if trim_context:
                candidate["context_frames"] = []
                candidate["adaptive_context_trimmed"] = True
            variants.append(candidate)
        return variants

    def request(
        self,
        path: str,
        payload: dict[str, Any] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if path not in {"/plan", "/audit"} or payload is None:
            return super().request(path, payload, **kwargs)

        last_error: BaseException | None = None
        variants = self._variants(path, payload)
        for index, candidate in enumerate(variants, start=1):
            try:
                result = super().request(path, candidate, **kwargs)
                if index > 1 and isinstance(result, dict):
                    result = dict(result)
                    result["adaptive_structured_output_recovery"] = {
                        "attempt": index,
                        "max_output_tokens": candidate.get("max_output_tokens"),
                        "reasoning_effort": candidate.get(
                            "planner_effort" if path == "/plan" else "auditor_effort"
                        ),
                        "context_trimmed": bool(candidate.get("adaptive_context_trimmed")),
                    }
                return result
            except RuntimeError as exc:
                if not self._is_structured_output_failure(exc):
                    raise
                last_error = exc

        stage = "planner" if path == "/plan" else "auditor"
        raise RuntimeError(
            f"OpenAI {stage} could not produce structured output after adaptive recovery "
            "with larger output budgets, lower reasoning effort, and reduced timeline context. "
            f"Last error: {last_error}"
        ) from last_error
