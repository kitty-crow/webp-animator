from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

DEFAULT_MODEL = os.environ.get("OPENAI_REPAIR_MODEL", "gpt-image-2")
DEFAULT_QUALITY = os.environ.get("OPENAI_REPAIR_QUALITY", "medium")
DEFAULT_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
DEFAULT_TIMEOUT = max(30.0, float(os.environ.get("OPENAI_REPAIR_TIMEOUT", "240")))

REPAIR_PROMPT = """You are repairing exactly one already-generated animation frame.

Image 1 is the candidate frame to edit. It is already at the intended temporal position and is the only image you may repair.
Image 2 is the immediately previous temporal anchor.
Image 3 is the immediately next temporal anchor.

Inspect Image 1 against Images 2 and 3. Correct defects in Image 1 such as malformed or duplicated structure, smearing, broken edges, inconsistent texture/details, bad occlusion reconstruction, or features that are visibly inconsistent with the two neighbouring anchors.

Do not choose a different moment in the motion. Preserve Image 1's pose, timing, camera, framing, scale, overall silhouette, colours, style and all unrelated content. Do not redesign the subject or scene. Make the smallest repair that makes the candidate coherent between its neighbours.

The application preserves transparency and may provide an edit mask. When a mask is supplied, treat its editable region as the user's explicit defect location and concentrate the correction there. Do not intentionally modify anything outside that marked region.
""".strip()


def status() -> dict:
    key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    return {
        "ready": bool(key),
        "model": DEFAULT_MODEL,
        "quality": DEFAULT_QUALITY,
        "base": DEFAULT_BASE,
    }


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def _selection_mask(value, size: tuple[int, int]) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        image = value.copy()
    else:
        path = Path(value)
        if not path.is_file():
            return None
        with Image.open(path) as source:
            image = source.copy()
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    if "A" in image.getbands():
        mask = image.getchannel("A")
    else:
        mask = image.convert("L")
    # Browser masks are stored as transparent background + opaque painted pixels.
    # Normalise every non-zero mark to an explicit 8-bit selection weight.
    return mask.point(lambda value: 255 if value > 0 else 0, mode="L")


def _api_edit_mask(selection: Image.Image) -> Image.Image:
    """Convert selected/painted pixels into the Images edit-mask convention.

    Transparent mask pixels are editable. Opaque mask pixels are preserved.
    """
    alpha = selection.point(lambda value: 0 if value > 0 else 255, mode="L")
    mask = Image.new("RGBA", selection.size, (255, 255, 255, 255))
    mask.putalpha(alpha)
    return mask


def _multipart(fields: Mapping[str, str], files: Sequence[tuple[str, str, str, bytes]]) -> tuple[bytes, str]:
    boundary = f"----webp-animator-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    crlf = b"\r\n"
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}".encode(),
                f'Content-Disposition: form-data; name="{name}"'.encode(),
                b"",
                str(value).encode("utf-8"),
            ]
        )
    for field, filename, mime, payload in files:
        chunks.extend(
            [
                f"--{boundary}".encode(),
                f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'.encode(),
                f"Content-Type: {mime}".encode(),
                b"",
                payload,
            ]
        )
    chunks.append(f"--{boundary}--".encode())
    chunks.append(b"")
    return crlf.join(chunks), f"multipart/form-data; boundary={boundary}"


def _error_message(error: urllib.error.HTTPError) -> str:
    try:
        raw = error.read().decode("utf-8", errors="replace")
    except Exception:
        raw = ""
    try:
        value = json.loads(raw)
        message = value.get("error", {}).get("message")
        if message:
            return str(message)
    except Exception:
        pass
    return raw.strip() or str(error)


def _image_edit(
    previous: Image.Image,
    candidate: Image.Image,
    following: Image.Image,
    *,
    selection: Image.Image | None,
    model: str,
    quality: str,
    prompt: str,
) -> tuple[Image.Image, dict]:
    key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("OpenAI repair is selected but OPENAI_API_KEY is not configured on the server.")

    candidate = candidate.convert("RGBA")
    size = candidate.size
    previous = previous.convert("RGBA")
    following = following.convert("RGBA")
    if previous.size != size or following.size != size:
        raise ValueError("OpenAI repair requires previous, candidate and next frames on the same canvas.")

    fields = {
        "model": model,
        "prompt": prompt,
        "quality": quality,
        "output_format": "png",
        "background": "transparent",
        "size": f"{size[0]}x{size[1]}",
    }
    files: list[tuple[str, str, str, bytes]] = [
        # The edit mask applies to the first input image, so candidate MUST be first.
        ("image[]", "candidate.png", "image/png", _png_bytes(candidate)),
        ("image[]", "previous-anchor.png", "image/png", _png_bytes(previous)),
        ("image[]", "next-anchor.png", "image/png", _png_bytes(following)),
    ]
    if selection is not None and selection.getbbox():
        files.append(("mask", "defect-mask.png", "image/png", _png_bytes(_api_edit_mask(selection))))

    body, content_type = _multipart(fields, files)
    request = urllib.request.Request(
        f"{DEFAULT_BASE}/images/edits",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": content_type,
            "Accept": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                value = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            message = _error_message(exc)
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise RuntimeError(f"OpenAI image repair failed ({exc.code}): {message}") from exc
            time.sleep(0.5 * (2**attempt))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == 2:
                raise RuntimeError(f"Could not reach OpenAI image repair endpoint: {exc}") from exc
            time.sleep(0.5 * (2**attempt))
    else:  # pragma: no cover
        raise RuntimeError(f"OpenAI image repair failed: {last_error}")

    item = (value.get("data") or [{}])[0]
    encoded = item.get("b64_json") or item.get("b64")
    if encoded:
        payload = base64.b64decode(encoded)
    elif item.get("url"):
        with urllib.request.urlopen(str(item["url"]), timeout=DEFAULT_TIMEOUT) as response:
            payload = response.read()
    else:
        raise RuntimeError("OpenAI image repair returned no image data.")

    with Image.open(io.BytesIO(payload)) as image:
        repaired = image.convert("RGBA").copy()
    if repaired.size != size:
        repaired = repaired.resize(size, Image.Resampling.LANCZOS)
    return repaired, value


def _constrain_result(candidate: Image.Image, repaired: Image.Image, selection: Image.Image | None) -> Image.Image:
    """Keep temporal geometry/alpha deterministic and enforce manual edit locality."""
    candidate = candidate.convert("RGBA")
    repaired = repaired.convert("RGBA")
    if repaired.size != candidate.size:
        repaired = repaired.resize(candidate.size, Image.Resampling.LANCZOS)

    # GPT never owns transparency. Reapply the deterministic candidate alpha exactly.
    candidate_alpha = candidate.getchannel("A")
    repaired.putalpha(candidate_alpha)

    if selection is not None and selection.getbbox():
        constrained = Image.composite(repaired, candidate, selection)
    else:
        constrained = repaired
    constrained.putalpha(candidate_alpha)
    return constrained


def repair_images(
    images: Sequence[Image.Image],
    target_indexes: Sequence[int],
    work_dir: Path,
    *,
    masks: Mapping[int, object] | None = None,
    loop: bool = False,
    progress=None,
    model: str | None = None,
    quality: str | None = None,
    prompt: str | None = None,
):
    frames = [image.convert("RGBA").copy() for image in images]
    targets = sorted({int(index) for index in target_indexes if 0 <= int(index) < len(frames)})
    masks = dict(masks or {})
    model = str(model or DEFAULT_MODEL)
    quality = str(quality or DEFAULT_QUALITY).lower()
    if quality not in {"low", "medium", "high"}:
        quality = "medium"
    prompt = str(prompt or REPAIR_PROMPT).strip()

    output_dir = Path(work_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_count = 0
    skipped: list[int] = []
    usage: list[dict] = []

    for ordinal, index in enumerate(targets, start=1):
        previous_index = index - 1
        next_index = index + 1
        if previous_index < 0:
            if loop and len(frames) > 2:
                previous_index = len(frames) - 1
            else:
                skipped.append(index)
                if progress:
                    progress(ordinal, len(targets))
                continue
        if next_index >= len(frames):
            if loop and len(frames) > 2:
                next_index = 0
            else:
                skipped.append(index)
                if progress:
                    progress(ordinal, len(targets))
                continue

        candidate = frames[index]
        selection = _selection_mask(masks.get(index), candidate.size)
        repaired, response = _image_edit(
            frames[previous_index],
            candidate,
            frames[next_index],
            selection=selection,
            model=model,
            quality=quality,
            prompt=prompt,
        )
        result = _constrain_result(candidate, repaired, selection)
        frames[index] = result
        result.save(output_dir / f"{index:06d}.png")
        repaired_count += 1
        if isinstance(response.get("usage"), dict):
            usage.append(response["usage"])
        if progress:
            progress(ordinal, len(targets))

    return frames, {
        "audited": len(targets),
        "repaired": repaired_count,
        "skipped": skipped,
        "engine": "openai",
        "model": model,
        "quality": quality,
        "manual_masks": sum(1 for index in targets if index in masks),
        "usage": usage,
        "alpha_policy": "candidate-alpha-authoritative",
    }
