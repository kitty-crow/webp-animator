from __future__ import annotations

import base64
import io
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageOps

import openai_repair as base

DEFAULT_MODEL = base.DEFAULT_MODEL
DEFAULT_QUALITY = base.DEFAULT_QUALITY
status = base.status

REPAIR_PROMPT = """You are a defect-repair tool for ONE already-generated animation frame. You are not an animator and you are not allowed to invent a new in-between frame.

IMAGE 1 is the candidate frame to edit. It already represents the intended moment in time. Preserve its pose, timing, camera, framing, scale, perspective, silhouette, colours, lighting and composition.
IMAGE 2 is the immediately previous temporal anchor.
IMAGE 3 is the immediately next temporal anchor.
A fourth image may contain the full animation in playback order and is CONTEXT ONLY.

Your only job is to remove or repaint visible GENERATION DEFECTS in IMAGE 1. Typical defects include doubled/ghost limbs or tails, smeared anatomy, duplicated edges, malformed hands/feet/facial details, broken clothing geometry, corrupted texture, bad occlusion reconstruction, or other obvious synthesis artefacts.

DO NOT add motion blur. DO NOT add ghost images. DO NOT create duplicate limbs, tails, ears, hands, feet, edges or silhouettes. DO NOT average the neighbouring poses together. DO NOT move the subject toward either anchor. DO NOT redesign or beautify the frame. DO NOT alter clean regions merely to make the rendering more consistent.

If the candidate already looks correct in an area, leave that area unchanged. Make the smallest local paint-over needed to remove the defect while keeping the candidate at exactly the same temporal state.

When an edit mask is supplied, the marked region is the user's explicit defect location. Repair only that marked region; anything outside it must remain unchanged. Transparency is owned by the application and must not be reinterpreted.
""".strip()


def _multiple_of_16(value: int) -> int:
    value = max(1, int(value))
    return ((value + 15) // 16) * 16


def _pad_rgba(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGBA")
    if image.size == size:
        return image.copy()
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.alpha_composite(image, (0, 0))
    return canvas


def _pad_selection(selection: Image.Image | None, size: tuple[int, int]) -> Image.Image | None:
    if selection is None:
        return None
    selection = selection.convert("L")
    if selection.size == size:
        return selection.copy()
    canvas = Image.new("L", size, 0)
    canvas.paste(selection, (0, 0))
    return canvas


def _checkerboard(size: tuple[int, int], cell: int = 16) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size, (232, 232, 232, 255))
    draw = ImageDraw.Draw(image)
    alt = (196, 196, 196, 255)
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            if ((x // cell) + (y // cell)) % 2:
                draw.rectangle((x, y, min(width - 1, x + cell - 1), min(height - 1, y + cell - 1)), fill=alt)
    return image


def _context_sheet(frames: Sequence[Image.Image], target_index: int) -> Image.Image:
    """Build one labelled visual containing every current animation frame."""
    count = max(1, len(frames))
    columns = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / columns))
    max_sheet = 2048
    cell = max(56, min(320, max_sheet // columns, max_sheet // rows))
    label_h = max(16, min(28, cell // 6))
    sheet = _checkerboard((columns * cell, rows * cell), max(8, cell // 12))
    draw = ImageDraw.Draw(sheet)

    previous_index = target_index - 1 if target_index > 0 else count - 1
    next_index = target_index + 1 if target_index < count - 1 else 0

    for index, source in enumerate(frames):
        col = index % columns
        row = index // columns
        x0 = col * cell
        y0 = row * cell
        inner = (max(1, cell - 8), max(1, cell - label_h - 8))
        thumb = ImageOps.contain(source.convert("RGBA"), inner, Image.Resampling.LANCZOS)
        left = x0 + (cell - thumb.width) // 2
        top = y0 + label_h + max(2, (cell - label_h - thumb.height) // 2)
        sheet.alpha_composite(thumb, (left, top))

        role = ""
        if index == target_index:
            role = " TARGET"
        elif index == previous_index:
            role = " PREV"
        elif index == next_index:
            role = " NEXT"
        draw.rectangle((x0, y0, x0 + cell - 1, y0 + label_h - 1), fill=(20, 20, 20, 230))
        draw.text((x0 + 4, y0 + 3), f"Frame {index + 1}{role}", fill=(255, 255, 255, 255))
        if role:
            draw.rectangle((x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2), outline=(255, 96, 96, 255), width=max(2, cell // 80))

    return _pad_rgba(sheet, (_multiple_of_16(sheet.width), _multiple_of_16(sheet.height)))


def _augment_prompt(prompt: str, target_index: int, frame_count: int) -> str:
    return (
        str(prompt).strip()
        + "\n\nIMAGE 4 is a labelled contact sheet containing the ENTIRE current animation in playback order. "
        + f"The edit target is Frame {target_index + 1} of {frame_count}. "
        + "Use IMAGE 4 only to understand persistent identity, proportions, clothing/details, motion direction, cadence and loop continuity. "
        + "IMAGE 2 and IMAGE 3 define the immediate temporal neighbourhood around IMAGE 1. "
        + "Do not borrow a cleaner pose from another frame. Do not interpolate again. Repair synthesis artefacts in IMAGE 1 only."
    )


def _image_edit(
    previous: Image.Image,
    candidate: Image.Image,
    following: Image.Image,
    *,
    context_frames: Sequence[Image.Image],
    target_index: int,
    selection: Image.Image | None,
    model: str,
    quality: str,
    prompt: str,
) -> tuple[Image.Image, dict]:
    key = base._api_key()
    if not key:
        raise RuntimeError("OpenAI repair is selected but OPENAI_API_KEY is not configured on the server or in the app .env file.")

    candidate = candidate.convert("RGBA")
    original_size = candidate.size
    previous = previous.convert("RGBA")
    following = following.convert("RGBA")
    if previous.size != original_size or following.size != original_size:
        raise ValueError("OpenAI repair requires previous, candidate and next frames on the same canvas.")

    api_size = (_multiple_of_16(original_size[0]), _multiple_of_16(original_size[1]))
    candidate_api = _pad_rgba(candidate, api_size)
    previous_api = _pad_rgba(previous, api_size)
    following_api = _pad_rgba(following, api_size)
    selection_api = _pad_selection(selection, api_size)
    context = _context_sheet(context_frames, target_index)

    fields = {
        "model": model,
        "prompt": _augment_prompt(prompt, target_index, len(context_frames)),
        "quality": quality,
        "output_format": "png",
        "background": "transparent",
        "size": f"{api_size[0]}x{api_size[1]}",
    }
    files: list[tuple[str, str, str, bytes]] = [
        ("image[]", "candidate.png", "image/png", base._png_bytes(candidate_api)),
        ("image[]", "previous-anchor.png", "image/png", base._png_bytes(previous_api)),
        ("image[]", "next-anchor.png", "image/png", base._png_bytes(following_api)),
        ("image[]", "whole-animation-context.png", "image/png", base._png_bytes(context)),
    ]
    if selection_api is not None and selection_api.getbbox():
        files.append(("mask", "defect-mask.png", "image/png", base._png_bytes(base._api_edit_mask(selection_api))))

    body, content_type = base._multipart(fields, files)
    request = urllib.request.Request(
        f"{base._base()}/images/edits",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=base._timeout()) as response:
                value = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            message = base._error_message(exc)
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
        with urllib.request.urlopen(str(item["url"]), timeout=base._timeout()) as response:
            payload = response.read()
    else:
        raise RuntimeError("OpenAI image repair returned no image data.")

    with Image.open(io.BytesIO(payload)) as image:
        repaired = image.convert("RGBA").copy()
    if repaired.size != api_size:
        repaired = repaired.resize(api_size, Image.Resampling.LANCZOS)
    return repaired.crop((0, 0, original_size[0], original_size[1])), value


def _change_fraction(candidate: Image.Image, repaired: Image.Image, threshold: int = 28) -> float:
    """Fraction of visible candidate pixels materially repainted by the model."""
    candidate = candidate.convert("RGBA")
    repaired = repaired.convert("RGBA")
    rgb_diff = ImageChops.difference(candidate.convert("RGB"), repaired.convert("RGB"))
    channels = rgb_diff.split()
    changed = Image.new("L", candidate.size, 0)
    cp = changed.load()
    rp, gp, bp = (channel.load() for channel in channels)
    alpha = candidate.getchannel("A").load()
    visible = 0
    changed_count = 0
    for y in range(candidate.height):
        for x in range(candidate.width):
            if alpha[x, y] <= 8:
                continue
            visible += 1
            if max(rp[x, y], gp[x, y], bp[x, y]) >= threshold:
                cp[x, y] = 255
                changed_count += 1
    return changed_count / max(1, visible)


def _constrain_and_guard(
    candidate: Image.Image,
    repaired: Image.Image,
    selection: Image.Image | None,
) -> tuple[Image.Image, dict]:
    constrained = base._constrain_result(candidate, repaired, selection)
    if selection is not None and selection.getbbox():
        return constrained, {"accepted": True, "guard": "manual-mask", "changed_fraction": None}

    # Auto repair is allowed to work without a user mask, but it is not allowed to
    # turn into a full-frame redraw. A broad repaint is safer to reject than to replace
    # a deterministic candidate with a newly hallucinated pose/ghosting pattern.
    fraction = _change_fraction(candidate, constrained)
    try:
        limit = float(base._setting("OPENAI_REPAIR_MAX_AUTO_CHANGE", "0.18"))
    except (TypeError, ValueError):
        limit = 0.18
    limit = max(0.02, min(0.75, limit))
    if fraction > limit:
        return candidate.convert("RGBA").copy(), {
            "accepted": False,
            "guard": "broad-repaint-rejected",
            "changed_fraction": round(fraction, 6),
            "max_changed_fraction": limit,
        }
    return constrained, {
        "accepted": True,
        "guard": "auto-locality",
        "changed_fraction": round(fraction, 6),
        "max_changed_fraction": limit,
    }


def repair_triplet(
    previous: Image.Image,
    candidate: Image.Image,
    following: Image.Image,
    *,
    context_frames: Sequence[Image.Image] | None = None,
    target_index: int = 1,
    mask=None,
    model: str | None = None,
    quality: str | None = None,
    prompt: str | None = None,
):
    candidate = candidate.convert("RGBA").copy()
    sequence = [image.convert("RGBA").copy() for image in (context_frames or [previous, candidate, following])]
    if not sequence:
        sequence = [previous.convert("RGBA"), candidate.copy(), following.convert("RGBA")]
        target_index = 1
    target_index = max(0, min(len(sequence) - 1, int(target_index)))
    selection = base._selection_mask(mask, candidate.size)
    repaired, response = _image_edit(
        previous,
        candidate,
        following,
        context_frames=sequence,
        target_index=target_index,
        selection=selection,
        model=str(model or base._setting("OPENAI_REPAIR_MODEL", DEFAULT_MODEL)),
        quality=str(quality or base._setting("OPENAI_REPAIR_QUALITY", DEFAULT_QUALITY)).lower(),
        prompt=str(prompt or REPAIR_PROMPT),
    )
    result, guard = _constrain_and_guard(candidate, repaired, selection)
    return result, {
        "engine": "openai",
        "model": str(model or base._setting("OPENAI_REPAIR_MODEL", DEFAULT_MODEL)),
        "quality": str(quality or base._setting("OPENAI_REPAIR_QUALITY", DEFAULT_QUALITY)).lower(),
        "manual_mask": bool(selection is not None and selection.getbbox()),
        "usage": response.get("usage") if isinstance(response, dict) else None,
        "alpha_policy": "candidate-alpha-authoritative",
        "context_policy": "whole-animation-contact-sheet",
        "context_frames": len(sequence),
        "api_canvas": [_multiple_of_16(candidate.width), _multiple_of_16(candidate.height)],
        "source_canvas": [candidate.width, candidate.height],
        **guard,
    }


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
    model = str(model or base._setting("OPENAI_REPAIR_MODEL", DEFAULT_MODEL))
    quality = str(quality or base._setting("OPENAI_REPAIR_QUALITY", DEFAULT_QUALITY)).lower()
    if quality not in {"low", "medium", "high"}:
        quality = "medium"
    prompt = str(prompt or REPAIR_PROMPT).strip()

    output_dir = Path(work_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_count = 0
    rejected_count = 0
    skipped: list[int] = []
    usage: list[dict] = []
    guards: list[dict] = []

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
        selection = base._selection_mask(masks.get(index), candidate.size)
        repaired, response = _image_edit(
            frames[previous_index],
            candidate,
            frames[next_index],
            context_frames=frames,
            target_index=index,
            selection=selection,
            model=model,
            quality=quality,
            prompt=prompt,
        )
        result, guard = _constrain_and_guard(candidate, repaired, selection)
        frames[index] = result
        result.save(output_dir / f"{index:06d}.png")
        guards.append(dict(index=index, **guard))
        if guard.get("accepted"):
            repaired_count += 1
        else:
            rejected_count += 1
        if isinstance(response.get("usage"), dict):
            usage.append(response["usage"])
        if progress:
            progress(ordinal, len(targets))

    source_size = frames[0].size if frames else (0, 0)
    return frames, {
        "audited": len(targets),
        "repaired": repaired_count,
        "rejected": rejected_count,
        "skipped": skipped,
        "engine": "openai",
        "model": model,
        "quality": quality,
        "manual_masks": sum(1 for index in targets if index in masks),
        "usage": usage,
        "guards": guards,
        "alpha_policy": "candidate-alpha-authoritative",
        "context_policy": "whole-animation-contact-sheet",
        "context_frames": len(frames),
        "api_canvas": [_multiple_of_16(source_size[0]), _multiple_of_16(source_size[1])] if frames else [0, 0],
        "source_canvas": [source_size[0], source_size[1]] if frames else [0, 0],
    }
