from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

CONTENT_MARGIN = 64


def read_manifest(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Worker manifest must be a JSON object")
    return value


def write_result(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_rgba(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA").copy()


def release_cuda(torch) -> None:
    """Drop Python garbage and return unused CUDA allocations to the allocator."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_cuda_oom(torch, exc: BaseException) -> bool:
    """Recognise CUDA OOMs across PyTorch versions without naming a missing class.

    Older PyTorch builds expose the exception as torch.cuda.OutOfMemoryError while
    newer builds may also expose torch.OutOfMemoryError. Referring directly to a
    missing top-level class inside an ``except`` clause raises AttributeError and
    masks the real CUDA OOM, so workers use this predicate instead.
    """
    classes = []
    for owner in (torch, getattr(torch, "cuda", None)):
        if owner is None:
            continue
        cls = getattr(owner, "OutOfMemoryError", None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            classes.append(cls)
    if classes and isinstance(exc, tuple(dict.fromkeys(classes))):
        return True

    if isinstance(exc, RuntimeError):
        text = str(exc).lower()
        return "out of memory" in text and (
            "cuda" in text or "cudnn" in text or "gpu" in text
        )
    return False


def content_bbox(first: Image.Image, second: Image.Image, margin: int = CONTENT_MARGIN):
    if first.size != second.size:
        raise ValueError("Engine input frames must share one canvas")
    union = ImageChops.lighter(first.getchannel("A"), second.getchannel("A"))
    bbox = union.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(first.width, right + margin),
        min(first.height, bottom + margin),
    )


def alpha_midpoint(first: Image.Image, second: Image.Image) -> Image.Image:
    a = np.asarray(first.getchannel("A"), dtype=np.float32)
    b = np.asarray(second.getchannel("A"), dtype=np.float32)
    alpha = np.clip(np.rint((a + b) * 0.5), 0, 255).astype(np.uint8)
    return Image.fromarray(alpha, "L")


def compose_rgb_with_alpha(rgb: Image.Image, alpha: Image.Image, canvas_size, bbox=None) -> Image.Image:
    rgb = rgb.convert("RGB")
    alpha = alpha.convert("L")
    rgba = Image.merge("RGBA", (*rgb.split(), alpha))
    if bbox is None and rgba.size == canvas_size:
        return rgba
    output = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if bbox is None:
        output.alpha_composite(rgba, (0, 0))
    else:
        output.alpha_composite(rgba, (bbox[0], bbox[1]))
    return output


def gap_score(first: Image.Image, second: Image.Image, alpha_threshold: int = 8) -> float:
    from smart_frames import _rgba_array, _score_arrays

    return _score_arrays(_rgba_array(first), _rgba_array(second), alpha_threshold)


def fixed_midpoint_fill(
    first: Image.Image,
    second: Image.Image,
    *,
    count: int,
    generate_midpoint,
    save_midpoint,
    alpha_threshold: int = 8,
):
    """Generate exactly ``count`` midpoint-only frames.

    Midpoint-only models cannot directly ask for arbitrary t. Split the currently
    longest temporal interval first, using visual gap score as a tie-breaker. This
    keeps the final temporal spacing as even as the model permits while preserving
    the exact requested frame count.
    """
    count = max(0, int(count))
    if count <= 0:
        return []

    segments = [
        {
            "t0": 0.0,
            "t1": 1.0,
            "left": first,
            "right": second,
            "score": gap_score(first, second, alpha_threshold),
        }
    ]
    generated = []

    for _ in range(count):
        index = max(
            range(len(segments)),
            key=lambda i: (
                segments[i]["t1"] - segments[i]["t0"],
                segments[i]["score"],
            ),
        )
        segment = segments.pop(index)
        tm = (segment["t0"] + segment["t1"]) * 0.5
        middle = generate_midpoint(segment["left"], segment["right"])
        ref = save_midpoint(middle, tm)
        generated.append((tm, ref))
        segments.extend(
            [
                {
                    "t0": segment["t0"],
                    "t1": tm,
                    "left": segment["left"],
                    "right": middle,
                    "score": gap_score(segment["left"], middle, alpha_threshold),
                },
                {
                    "t0": tm,
                    "t1": segment["t1"],
                    "left": middle,
                    "right": segment["right"],
                    "score": gap_score(middle, segment["right"], alpha_threshold),
                },
            ]
        )

    generated.sort(key=lambda item: item[0])
    return [ref for _, ref in generated]


def adaptive_midpoint_fill(
    first: Image.Image,
    second: Image.Image,
    *,
    threshold: float,
    max_frames: int,
    generate_midpoint,
    save_midpoint,
    alpha_threshold: int = 8,
):
    """Fill the worst remaining sub-gap until every gap meets the threshold.

    ``max_frames`` is a safety ceiling for automatic mode. The result contains the
    ordered saved-frame refs plus diagnostics so the caller can report whether the
    requested threshold was actually achieved.
    """
    max_frames = max(0, int(max_frames))
    threshold = float(threshold)
    segments = [
        {
            "t0": 0.0,
            "t1": 1.0,
            "left": first,
            "right": second,
            "score": gap_score(first, second, alpha_threshold),
        }
    ]
    generated = []

    while len(generated) < max_frames:
        worst_index = max(range(len(segments)), key=lambda i: segments[i]["score"])
        worst = segments[worst_index]
        if float(worst["score"]) <= threshold:
            break

        segment = segments.pop(worst_index)
        tm = (segment["t0"] + segment["t1"]) * 0.5
        middle = generate_midpoint(segment["left"], segment["right"])
        ref = save_midpoint(middle, tm)
        generated.append((tm, ref))
        segments.extend(
            [
                {
                    "t0": segment["t0"],
                    "t1": tm,
                    "left": segment["left"],
                    "right": middle,
                    "score": gap_score(segment["left"], middle, alpha_threshold),
                },
                {
                    "t0": tm,
                    "t1": segment["t1"],
                    "left": middle,
                    "right": segment["right"],
                    "score": gap_score(middle, segment["right"], alpha_threshold),
                },
            ]
        )

    generated.sort(key=lambda item: item[0])
    max_score = max((float(segment["score"]) for segment in segments), default=0.0)
    return {
        "frames": [ref for _, ref in generated],
        "satisfied": max_score <= threshold,
        "max_score": max_score,
        "limit_reached": len(generated) >= max_frames and max_score > threshold,
    }


def recursive_midpoints(
    first: Image.Image,
    second: Image.Image,
    *,
    depth: int,
    threshold: float | None,
    generate_midpoint,
    save_midpoint,
    alpha_threshold: int = 8,
    t0: float = 0.0,
    t1: float = 1.0,
):
    """Generate ordered recursive midpoints while retaining their exact temporal t."""
    if depth <= 0:
        return []
    if threshold is not None and gap_score(first, second, alpha_threshold) <= threshold:
        return []

    tm = (t0 + t1) * 0.5
    middle = generate_midpoint(first, second)
    middle_ref = save_midpoint(middle, tm)
    if depth == 1:
        return [middle_ref]

    left = recursive_midpoints(
        first,
        middle,
        depth=depth - 1,
        threshold=threshold,
        generate_midpoint=generate_midpoint,
        save_midpoint=save_midpoint,
        alpha_threshold=alpha_threshold,
        t0=t0,
        t1=tm,
    )
    right = recursive_midpoints(
        middle,
        second,
        depth=depth - 1,
        threshold=threshold,
        generate_midpoint=generate_midpoint,
        save_midpoint=save_midpoint,
        alpha_threshold=alpha_threshold,
        t0=tm,
        t1=t1,
    )
    return left + [middle_ref] + right
