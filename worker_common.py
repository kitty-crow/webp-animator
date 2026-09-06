from __future__ import annotations

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
