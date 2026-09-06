from __future__ import annotations

import math

from PIL import Image

from anim_align_webp import ShiftResult, find_best_translation


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


def _proxy_match(
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
    proxy_shift = max(0, round(max_shift * world_scale))
    shift = find_best_translation(
        pa,
        pb,
        axis=axis,
        max_shift_x=proxy_shift,
        max_shift_y=proxy_shift,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
        # The inputs are already proxies. This therefore performs only a tiny
        # exact refinement on the proxy instead of thousands of full-size scans.
        proxy_max_side=max(pa.width, pa.height, pb.width, pb.height),
    )
    return shift, world_scale


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

    The old implementation resized the full-resolution frame for every candidate
    scale and then asked the translation matcher to perform a full-resolution
    refinement for every candidate. On 2K/3K frames that can mean tens of
    thousands of multi-megapixel similarity evaluations per animation frame.

    Scale candidates are now evaluated entirely on a common low-resolution world
    proxy. Only the winning scale is rendered at source resolution. Translation
    is then refined once on a larger proxy and mapped back to source pixels.
    """
    if axis == "none":
        # Scale still matters even when panning is disabled.
        max_shift = 0

    low, high, guesses = _scale_search_bounds(anchor, image)

    coarse_count = 13
    coarse_step = (high - low) / max(1, coarse_count - 1)
    candidates = {low + coarse_step * i for i in range(coarse_count)}
    candidates.update(max(low, min(high, guess)) for guess in guesses)

    best: tuple[float, float, float] | None = None

    def consider(scale: float, max_side: int) -> None:
        nonlocal best
        shift, _ = _proxy_match(
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
        candidate = (shift.score, -abs(scale - 1.0), scale)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    for scale in sorted(candidates):
        consider(scale, 192)

    assert best is not None
    best_scale = best[2]
    refine_radius = max(coarse_step, 0.015)
    refine_low = max(low, best_scale - refine_radius)
    refine_high = min(high, best_scale + refine_radius)
    refine_count = 11
    refine_step = (refine_high - refine_low) / max(1, refine_count - 1)

    for i in range(refine_count):
        consider(refine_low + refine_step * i, 320)

    final_scale = best[2]
    final_image = _uniform_resize(image, final_scale)

    # One high-resolution proxy pass is enough after scale selection. 1280px
    # keeps this practical on large source frames while normally resolving the
    # final translation to within about one source pixel.
    final_proxy, world_scale = _proxy_match(
        anchor,
        image,
        object_scale=final_scale,
        max_object_scale=final_scale,
        axis=axis,
        max_shift=max_shift,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
        max_side=1280,
    )

    if axis == "none":
        final_shift = ShiftResult(0, 0, final_proxy.score)
    else:
        dx = round(final_proxy.dx / world_scale) if world_scale else 0
        dy = round(final_proxy.dy / world_scale) if world_scale else 0
        dx = max(-max_shift, min(max_shift, dx))
        dy = max(-max_shift, min(max_shift, dy))
        if axis == "x":
            dy = 0
        elif axis == "y":
            dx = 0
        final_shift = ShiftResult(dx, dy, final_proxy.score)

    return final_image, final_scale, final_shift
