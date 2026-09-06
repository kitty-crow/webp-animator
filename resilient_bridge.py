from __future__ import annotations

import base64
import io
import threading
from copy import deepcopy
from typing import Any, Callable

from PIL import Image, ImageDraw

from openai_job_features import AuditorAwareBridgeClient


_STRUCTURED_OUTPUT_MARKERS = (
    "Could not parse the expected output",
    "OpenAI returned no output text",
    "Structured output was unavailable",
)


class OpenAIJobCancelled(RuntimeError):
    """Raised between paid stages after the user has stopped a job."""


def _decode_rgba(value: str | None) -> Image.Image | None:
    if not value:
        return None
    try:
        with Image.open(io.BytesIO(base64.b64decode(value))) as source:
            return source.convert("RGBA")
    except Exception:
        return None


def _has_transparency(value: str | None) -> bool:
    image = _decode_rgba(value)
    if image is None:
        return False
    lo, _hi = image.getchannel("A").getextrema()
    return lo < 250


def _references_expect_transparency(payload: dict[str, Any]) -> bool:
    refs = list(payload.get("references", []) or [])
    if refs:
        return any(_has_transparency(str(ref.get("image_b64", ""))) for ref in refs if isinstance(ref, dict))
    return _has_transparency(str(payload.get("frame_a_b64", ""))) or _has_transparency(str(payload.get("frame_b_b64", "")))


def _repair_flat_opaque_background(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Recover alpha when an image edit paints a flat background despite transparent anchors.

    GPT Image is explicitly asked for transparent output, but image models can occasionally
    return a fully opaque matte. Only repair that narrow case: the anchors must actually
    contain alpha, the candidate must be effectively opaque, and at least three corners
    must agree on a flat background colour. Edge-connected pixels near that colour are
    then made transparent. Interior dark details are not globally keyed out.
    """
    if not payload.get("transparent") or not _references_expect_transparency(payload):
        return result
    encoded = result.get("image_b64")
    image = _decode_rgba(str(encoded or ""))
    if image is None:
        return result
    alpha = image.getchannel("A")
    lo, hi = alpha.getextrema()
    if lo < 245 or hi < 250:
        return result

    width, height = image.size
    if width < 2 or height < 2:
        return result
    coordinates = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    colours = [image.getpixel(point)[:3] for point in coordinates]
    median = tuple(sorted(channel)[len(channel) // 2] for channel in zip(*colours))

    def close(colour: tuple[int, int, int], tolerance: int = 34) -> bool:
        return max(abs(int(colour[index]) - int(median[index])) for index in range(3)) <= tolerance

    matching = [point for point, colour in zip(coordinates, colours) if close(colour)]
    if len(matching) < 3:
        return result

    repaired = image.copy()
    for point in matching:
        red, green, blue, _alpha = repaired.getpixel(point)
        ImageDraw.floodfill(repaired, point, (red, green, blue, 0), thresh=34)

    repaired_alpha = repaired.getchannel("A")
    repaired_lo, _ = repaired_alpha.getextrema()
    if repaired_lo >= 245:
        return result

    buffer = io.BytesIO()
    repaired.save(buffer, format="PNG")
    value = dict(result)
    value["image_b64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
    value["alpha_repaired"] = True
    value["alpha_repair_reason"] = "Transparent source animation received a flat opaque edge-connected matte from the image model."
    return value


def _correctable_violation(item: dict[str, Any], transparent_expected: bool) -> bool:
    text = " ".join(str(item.get(key, "")) for key in ("type", "region", "description")).casefold()

    if transparent_expected and "background" in text and any(
        word in text for word in ("black", "colour", "color", "opaque", "transparent", "canvas", "matte")
    ):
        return True

    global_subject = any(
        word in text for word in (
            "whole frame", "whole image", "entire frame", "entire image", "entire subject",
            "overall", "global", "canvas", "framing", "centred", "centered",
        )
    )
    translation = any(word in text for word in ("shifted", "translation", "translated", "offset", "too high", "too low", "upward", "downward"))
    scaling = any(word in text for word in ("uniform scale", "scaled up", "scaled down", "larger", "smaller", "overall size"))
    structural = any(word in text for word in ("limb", "hand", "foot", "arm", "leg", "pose", "anatom", "occlusion", "topology", "missing", "extra"))

    if translation and (global_subject or not structural):
        return True
    if scaling and (global_subject or not structural):
        return True
    return False


def _relax_pipeline_correctable_audit(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    audit = result.get("audit")
    if not isinstance(audit, dict) or audit.get("acceptable"):
        return result

    transparent_expected = _references_expect_transparency(payload)
    violations = [item for item in list(audit.get("violations", []) or []) if isinstance(item, dict)]
    correctable = [item for item in violations if _correctable_violation(item, transparent_expected)]
    substantive = [item for item in violations if item not in correctable]
    if not correctable:
        return result

    amended = dict(audit)
    amended["violations"] = substantive
    observations = list(amended.get("observations", []) or [])
    observations.append(
        "Pipeline tolerance applied: global translation/uniform scale and transparent-canvas presentation are corrected downstream and are not semantic interpolation failures."
    )
    amended["observations"] = observations
    amended["pipeline_correctable_violations"] = correctable
    if not substantive:
        amended["acceptable"] = True
        amended["retry_recommended"] = False
        amended["accepted_by_pipeline_tolerance"] = True

    value = dict(result)
    value["audit"] = amended
    return value


class ResilientAuditorAwareBridgeClient(AuditorAwareBridgeClient):
    """OpenAI bridge with adaptive recovery plus alpha/geometry pipeline awareness."""

    def __init__(self, root, port: int = 18744):
        super().__init__(root, port)
        self._job_context = threading.local()

    def set_current_job(self, job_id: str, cancelled: Callable[[str], bool]) -> None:
        self._job_context.job_id = job_id
        self._job_context.cancelled = cancelled

    def clear_current_job(self) -> None:
        self._job_context.job_id = None
        self._job_context.cancelled = None

    def _raise_if_cancelled(self) -> None:
        job_id = getattr(self._job_context, "job_id", None)
        checker = getattr(self._job_context, "cancelled", None)
        if job_id and callable(checker) and checker(str(job_id)):
            raise OpenAIJobCancelled(f"OpenAI job {job_id} was stopped by the user.")

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

    @staticmethod
    def _pipeline_payload(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = deepcopy(payload)
        transparent_expected = _references_expect_transparency(candidate)

        if path == "/generate" and candidate.get("transparent"):
            original = str(candidate.get("generation_instruction", "")).strip()
            alpha_instruction = (
                "OUTPUT-CANVAS REQUIREMENT: this animation uses a true transparent alpha canvas. "
                "Everything outside the depicted foreground content must remain alpha=0. Do not paint a black, white, coloured, checkerboard, or other background. "
                "A dark/black-looking surround in a reference preview may simply be the viewer compositing transparent pixels; it is not background artwork to reproduce."
            )
            candidate["generation_instruction"] = f"{alpha_instruction}\n\n{original}" if original else alpha_instruction

        if path in {"/plan", "/audit"} and transparent_expected:
            original = str(candidate.get("user_instruction", "")).strip()
            alpha_context = (
                "PIPELINE FACT: the supplied source animation has a transparent alpha background. Blank/dark/black-looking canvas around the foreground is transparency, not coloured background graphics. "
                "Do not describe preserving or replacing such pixels as background artwork."
            )
            candidate["user_instruction"] = f"{original}\n\n{alpha_context}" if original else alpha_context

        if path == "/audit":
            original = str(candidate.get("user_instruction", "")).strip()
            geometry_context = (
                "AUDIT SCOPE FOR THIS PIPELINE: judge semantic interpolation, pose/state, structure, topology, occlusion and genuinely changed visual content. "
                "Do NOT reject merely because the whole generated foreground is shifted up/down/left/right, is uniformly a little larger/smaller, is centred differently, or needs canvas-fit correction. "
                "The next deterministic WebP Animator stage explicitly optimises uniform resize and X/Y translation to maximise pixel compatibility. Those are correctable presentation differences, not failed interpolation."
            )
            candidate["user_instruction"] = f"{original}\n\n{geometry_context}" if original else geometry_context

        return candidate

    def request(self, path: str, payload: dict[str, Any] | None, **kwargs: Any) -> dict[str, Any]:
        self._raise_if_cancelled()
        if payload is None:
            result = super().request(path, payload, **kwargs)
            self._raise_if_cancelled()
            return result

        prepared = self._pipeline_payload(path, payload)
        if path not in {"/plan", "/audit"}:
            result = super().request(path, prepared, **kwargs)
            self._raise_if_cancelled()
            if path == "/generate" and isinstance(result, dict):
                result = _repair_flat_opaque_background(result, prepared)
            return result

        last_error: BaseException | None = None
        variants = self._variants(path, prepared)
        for index, candidate in enumerate(variants, start=1):
            self._raise_if_cancelled()
            try:
                result = super().request(path, candidate, **kwargs)
                self._raise_if_cancelled()
                if path == "/audit" and isinstance(result, dict):
                    result = _relax_pipeline_correctable_audit(result, candidate)
                if index > 1 and isinstance(result, dict):
                    result = dict(result)
                    result["adaptive_structured_output_recovery"] = {
                        "attempt": index,
                        "max_output_tokens": candidate.get("max_output_tokens"),
                        "reasoning_effort": candidate.get("planner_effort" if path == "/plan" else "auditor_effort"),
                        "context_trimmed": bool(candidate.get("adaptive_context_trimmed")),
                    }
                return result
            except OpenAIJobCancelled:
                raise
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
