from __future__ import annotations

import math

import numpy as np
from PIL import Image

from anim_align_webp import ShiftResult, translation_score


def _uniform_resize(image: Image.Image, scale: float) -> Image.Image:
    if abs(scale - 1.0) < 1e-9:
        return image
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _scale_search_bounds(anchor: Image.Image, image: Image.Image):
    width_ratio = anchor.width / max(1, image.width)
    height_ratio = anchor.height / max(1, image.height)
    area_ratio = ((anchor.width * anchor.height) / max(1, image.width * image.height)) ** 0.5
    guesses = [1.0, width_ratio, height_ratio, area_ratio]
    low = max(0.1, min(guesses) * 0.7)
    high = min(3.0, max(guesses) * 1.3)
    if high <= low:
        high = min(3.0, low + 0.1)
    return low, high, guesses


def _resize_world(image: Image.Image, world_scale: float, object_scale: float = 1.0) -> Image.Image:
    width = max(1, round(image.width * world_scale * object_scale))
    height = max(1, round(image.height * world_scale * object_scale))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _proxy_pair(
    anchor: Image.Image,
    image: Image.Image,
    *,
    object_scale: float,
    max_side: int,
    max_object_scale: float,
):
    longest = max(
        anchor.width,
        anchor.height,
        image.width * max_object_scale,
        image.height * max_object_scale,
        1,
    )
    world_scale = min(1.0, max_side / longest)
    return (
        _resize_world(anchor, world_scale),
        _resize_world(image, world_scale, object_scale),
        world_scale,
    )


def _normaliser(a: np.ndarray, b: np.ndarray, alpha_threshold: int) -> int:
    return max(
        int(np.count_nonzero(a[..., 3] > alpha_threshold)),
        int(np.count_nonzero(b[..., 3] > alpha_threshold)),
        1,
    )


def _score_grid(
    a: np.ndarray,
    b: np.ndarray,
    *,
    axis: str,
    max_x: int,
    max_y: int,
    sigma: float,
    alpha_threshold: int,
    centre_x: int = 0,
    centre_y: int = 0,
    radius_x: int | None = None,
    radius_y: int | None = None,
    step: int = 1,
) -> ShiftResult:
    if axis == "none":
        return ShiftResult(
            0,
            0,
            translation_score(
                a,
                b,
                0,
                0,
                sigma=sigma,
                alpha_threshold=alpha_threshold,
                normaliser=_normaliser(a, b, alpha_threshold),
            ),
        )

    if axis == "x":
        max_y = 0
    elif axis == "y":
        max_x = 0

    if radius_x is None:
        x0, x1 = -max_x, max_x
    else:
        x0 = max(-max_x, centre_x - radius_x)
        x1 = min(max_x, centre_x + radius_x)
    if radius_y is None:
        y0, y1 = -max_y, max_y
    else:
        y0 = max(-max_y, centre_y - radius_y)
        y1 = min(max_y, centre_y + radius_y)

    if axis == "x":
        y0 = y1 = 0
    elif axis == "y":
        x0 = x1 = 0

    normaliser = _normaliser(a, b, alpha_threshold)
    best = ShiftResult(0, 0, -math.inf)
    for dy in range(y0, y1 + 1, max(1, step)):
        for dx in range(x0, x1 + 1, max(1, step)):
            score = translation_score(
                a,
                b,
                dx,
                dy,
                sigma=sigma,
                alpha_threshold=alpha_threshold,
                normaliser=normaliser,
            )
            if score > best.score:
                best = ShiftResult(dx, dy, score)
    return best


def _global_proxy_match(
    anchor: Image.Image,
    image: Image.Image,
    *,
    object_scale: float,
    max_object_scale: float,
    axis: str,
    max_shift: int,
    sigma: float,
    alpha_threshold: int,
    max_side: int,
) -> tuple[ShiftResult, float]:
    pa, pb, world_scale = _proxy_pair(
        anchor,
        image,
        object_scale=object_scale,
        max_side=max_side,
        max_object_scale=max_object_scale,
    )
    a = np.asarray(pa)
    b = np.asarray(pb)
    proxy_shift = max(0, round(max_shift * world_scale))
    largest = proxy_shift if axis != "none" else 0
    step = max(1, math.ceil(largest / 10))

    coarse = _score_grid(
        a,
        b,
        axis=axis,
        max_x=proxy_shift,
        max_y=proxy_shift,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
        step=step,
    )
    if step == 1 or axis == "none":
        return coarse, world_scale

    refined = _score_grid(
        a,
        b,
        axis=axis,
        max_x=proxy_shift,
        max_y=proxy_shift,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
        centre_x=coarse.dx,
        centre_y=coarse.dy,
        radius_x=step,
        radius_y=step,
        step=1,
    )
    return refined, world_scale


def _local_proxy_match(
    anchor: Image.Image,
    image: Image.Image,
    *,
    object_scale: float,
    axis: str,
    max_shift: int,
    sigma: float,
    alpha_threshold: int,
    max_side: int,
    guess_dx: int,
    guess_dy: int,
    radius: int,
) -> tuple[ShiftResult, float]:
    pa, pb, world_scale = _proxy_pair(
        anchor,
        image,
        object_scale=object_scale,
        max_side=max_side,
        max_object_scale=object_scale,
    )
    a = np.asarray(pa)
    b = np.asarray(pb)
    proxy_shift = max(0, round(max_shift * world_scale))
    centre_x = round(guess_dx * world_scale)
    centre_y = round(guess_dy * world_scale)
    result = _score_grid(
        a,
        b,
        axis=axis,
        max_x=proxy_shift,
        max_y=proxy_shift,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
        centre_x=centre_x,
        centre_y=centre_y,
        radius_x=radius,
        radius_y=radius,
        step=1,
    )
    return result, world_scale


def _to_source_shift(shift: ShiftResult, world_scale: float, max_shift: int, axis: str):
    if axis == "none":
        return 0, 0
    dx = round(shift.dx / world_scale) if world_scale else 0
    dy = round(shift.dy / world_scale) if world_scale else 0
    dx = max(-max_shift, min(max_shift, dx))
    dy = max(-max_shift, min(max_shift, dy))
    if axis == "x":
        dy = 0
    elif axis == "y":
        dx = 0
    return dx, dy


def find_best_scale_and_translation(
    anchor: Image.Image,
    image: Image.Image,
    *,
    axis: str,
    max_shift: int,
    sigma: float,
    alpha_threshold: int,
):
    """Fast coarse-to-fine scale + pan match against frame 1.

    Candidate scales are evaluated only on small common-coordinate proxies. The
    winning scale is rendered once at source resolution, while translation is
    successively refined on 384px and 1280px proxies around the previous estimate.
    This avoids the old O(scales × full-resolution XY search) behaviour that made
    2K/3K animations appear frozen for many minutes.
    """
    if axis == "none":
        max_shift = 0

    low, high, guesses = _scale_search_bounds(anchor, image)
    coarse_count = 11
    coarse_step = (high - low) / max(1, coarse_count - 1)
    candidates = {low + coarse_step * i for i in range(coarse_count)}
    candidates.update(max(low, min(high, guess)) for guess in guesses)

    best: tuple[float, float, float, int, int] | None = None

    def consider(scale: float, max_side: int) -> None:
        nonlocal best
        shift, world_scale = _global_proxy_match(
            anchor,
            image,
            object_scale=scale,
            max_object_scale=high,
            axis=axis,
            max_shift=max_shift,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
            max_side=max_side,
        )
        dx, dy = _to_source_shift(shift, world_scale, max_shift, axis)
        candidate = (shift.score, -abs(scale - 1.0), scale, dx, dy)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    for scale in sorted(candidates):
        consider(scale, 96)

    assert best is not None
    best_scale = best[2]
    refine_radius = max(coarse_step, 0.015)
    refine_low = max(low, best_scale - refine_radius)
    refine_high = min(high, best_scale + refine_radius)
    refine_count = 9
    refine_step = (refine_high - refine_low) / max(1, refine_count - 1)

    for i in range(refine_count):
        consider(refine_low + refine_step * i, 160)

    final_scale = best[2]
    guess_dx, guess_dy = best[3], best[4]

    # Refine only around the current estimate. The scale-search proxy is already
    # exact to about one low-res pixel, so a small local window at each larger
    # pyramid level is enough and does not rescan the whole pan range.
    mid, mid_world = _local_proxy_match(
        anchor,
        image,
        object_scale=final_scale,
        axis=axis,
        max_shift=max_shift,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
        max_side=384,
        guess_dx=guess_dx,
        guess_dy=guess_dy,
        radius=4,
    )
    guess_dx, guess_dy = _to_source_shift(mid, mid_world, max_shift, axis)

    high_result, high_world = _local_proxy_match(
        anchor,
        image,
        object_scale=final_scale,
        axis=axis,
        max_shift=max_shift,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
        max_side=1280,
        guess_dx=guess_dx,
        guess_dy=guess_dy,
        radius=3,
    )
    final_dx, final_dy = _to_source_shift(high_result, high_world, max_shift, axis)

    final_image = _uniform_resize(image, final_scale)
    final_shift = ShiftResult(final_dx, final_dy, high_result.score)
    return final_image, final_scale, final_shift
