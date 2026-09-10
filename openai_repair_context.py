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

from PIL import Image, ImageDraw, ImageOps

import openai_repair as base

DEFAULT_MODEL = base.DEFAULT_MODEL
DEFAULT_QUALITY = base.DEFAULT_QUALITY
REPAIR_PROMPT = base.REPAIR_PROMPT
status = base.status


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
    """Build one labelled visual containing every current animation frame.

    Keeping the entire sequence in a single context image avoids arbitrary image-count
    limits while still giving the image model global motion/style context for each
    local repair call.
    """
    count = max(1, len(frames))
    columns = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / columns))
    max_sheet = 2048
    cell = max(56, min(320, max_sheet // columns, max_sheet // rows))
    label_h = max(16, min(28, cell // 6))
    sheet = _checkerboard((columns * cell, rows * cell), max(8, cell // 12))
    draw = ImageDraw.Draw(sheet)

    previous_index = target_index - 1
    next_index = target_index + 1
    if target_index == 0:
        previous_index = count - 1
    if target_index == count - 1:
        next_index = 0

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

    padded = (_multiple_of_16(sheet.width), _multiple_of_16(sheet.height))
    return _pad_rgba(sheet, padded)


def _augment_prompt(prompt: str, target_index: int, frame_count: int) -> str:
    return (
        str(prompt).strip()
        + "\n\nImage 4 is a labelled contact sheet containing the ENTIRE current animation in playback order. "
        + f"The edit target is Frame {target_index + 1} of {frame_count}. "
        + "Use Image 4 as global context for identity, proportions, style, motion direction, cadence, recurring details and loop continuity. "
        + "Images 2 and 3 are still the immediate temporal anchors around Image 1, so they control the local in-between state. "
        + "Do not invent a different temporal pose merely because another frame in the global context looks cleaner."
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
    repaired = repaired.crop((0, 0, original_size[0], original_size[1]))
    return repaired, value


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
    result = base._constrain_result(candidate, repaired, selection)
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
        result = base._constrain_result(candidate, repaired, selection)
        frames[index] = result
        result.save(output_dir / f"{index:06d}.png")
        repaired_count += 1
        if isinstance(response.get("usage"), dict):
            usage.append(response["usage"])
        if progress:
            progress(ordinal, len(targets))

    source_size = frames[0].size if frames else (0, 0)
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
        "context_policy": "whole-animation-contact-sheet",
        "context_frames": len(frames),
        "api_canvas": [_multiple_of_16(source_size[0]), _multiple_of_16(source_size[1])] if frames else [0, 0],
        "source_canvas": [source_size[0], source_size[1]] if frames else [0, 0],
    }
