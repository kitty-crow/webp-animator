from __future__ import annotations

import math
import os
from typing import Iterable

import numpy as np
from PIL import Image


def _torch_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


def cuda_available() -> bool:
    return _torch_cuda() is not None


def _candidate_ranges(axis: str, max_x: int, max_y: int, step: int):
    step = max(1, step)
    if axis == "x":
        return range(-max_x, max_x + 1, step), (0,)
    if axis == "y":
        return (0,), range(-max_y, max_y + 1, step)
    if axis == "none":
        return (0,), (0,)
    return range(-max_x, max_x + 1, step), range(-max_y, max_y + 1, step)


def _active_count(array: np.ndarray, alpha_threshold: int) -> int:
    active = int(np.count_nonzero(array[..., 3] > alpha_threshold))
    return active if active > 0 else array.shape[0] * array.shape[1]


def _score_candidates_gpu(
    a_img: Image.Image,
    b_img: Image.Image,
    candidates: list[tuple[int, int]],
    *,
    sigma: float,
    alpha_threshold: int,
):
    """Evaluate many exact integer translations in parallel on CUDA.

    The scoring formula matches anim_align_webp.translation_score: soft RGBA
    similarity is accumulated only inside the rectangular overlap and divided by
    the larger meaningful-pixel count. Candidate shifts are chunked to keep VRAM
    bounded even when the source canvas is large.
    """
    torch = _torch_cuda()
    if torch is None:
        raise RuntimeError("CUDA PyTorch is not available")
    if not candidates:
        raise ValueError("No translation candidates")

    a_np = np.asarray(a_img.convert("RGBA"), dtype=np.uint8)
    b_np = np.asarray(b_img.convert("RGBA"), dtype=np.uint8)
    ha, wa = a_np.shape[:2]
    hb, wb = b_np.shape[:2]

    min_dx = min(dx for dx, _ in candidates)
    max_dx = max(dx for dx, _ in candidates)
    min_dy = min(dy for _, dy in candidates)
    max_dy = max(dy for _, dy in candidates)
    world_min_x = min(0, min_dx)
    world_min_y = min(0, min_dy)
    world_max_x = max(wa, max_dx + wb)
    world_max_y = max(ha, max_dy + hb)
    canvas_w = max(1, world_max_x - world_min_x)
    canvas_h = max(1, world_max_y - world_min_y)
    a_x = -world_min_x
    a_y = -world_min_y

    device = torch.device("cuda")
    a = torch.zeros((4, canvas_h, canvas_w), device=device, dtype=torch.float32)
    a_occ = torch.zeros((canvas_h, canvas_w), device=device, dtype=torch.bool)
    a_src = torch.from_numpy(a_np).to(device=device, dtype=torch.float32).permute(2, 0, 1)
    a[:, a_y:a_y + ha, a_x:a_x + wa] = a_src
    a_occ[a_y:a_y + ha, a_x:a_x + wa] = True

    b_src = torch.from_numpy(b_np).to(device=device, dtype=torch.float32).permute(2, 0, 1)
    normaliser = max(_active_count(a_np, alpha_threshold), _active_count(b_np, alpha_threshold), 1)

    bytes_per_candidate = max(1, canvas_h * canvas_w * (4 * 4 + 1 + 4 + 1))
    target_bytes = int(os.environ.get("FRAME_MATCH_GPU_BYTES", 256 * 1024 * 1024))
    chunk_size = max(1, min(64, target_bytes // bytes_per_candidate))
    best_score = -math.inf
    best_dx = 0
    best_dy = 0

    try:
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start:start + chunk_size]
            count = len(chunk)
            b_batch = torch.zeros((count, 4, canvas_h, canvas_w), device=device, dtype=torch.float32)
            b_occ = torch.zeros((count, canvas_h, canvas_w), device=device, dtype=torch.bool)
            for row, (dx, dy) in enumerate(chunk):
                bx = dx - world_min_x
                by = dy - world_min_y
                b_batch[row, :, by:by + hb, bx:bx + wb] = b_src
                b_occ[row, by:by + hb, bx:bx + wb] = True

            overlap = b_occ & a_occ.unsqueeze(0)
            a_batch = a.unsqueeze(0)
            active = overlap & (
                (a_batch[:, 3] > alpha_threshold) | (b_batch[:, 3] > alpha_threshold)
            )
            dist = (a_batch - b_batch).abs().mean(dim=1)
            similarity = torch.exp(-torch.square(dist / float(sigma)))
            similarity = torch.where(active, similarity, torch.zeros_like(similarity))
            scores = similarity.sum(dim=(1, 2), dtype=torch.float64) / float(normaliser)
            value, row = torch.max(scores, dim=0)
            score = float(value.item())
            if score > best_score:
                best_score = score
                best_dx, best_dy = chunk[int(row.item())]

            del scores, similarity, dist, active, overlap, b_occ, b_batch

        from anim_align_webp import ShiftResult

        return ShiftResult(best_dx, best_dy, best_score)
    finally:
        del b_src, a_src, a_occ, a
        torch.cuda.empty_cache()


def _search_gpu(
    a_img: Image.Image,
    b_img: Image.Image,
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
):
    if axis == "none":
        candidates = [(0, 0)]
    else:
        if axis == "x":
            max_y = 0
        elif axis == "y":
            max_x = 0
        x0 = -max_x if radius_x is None else max(-max_x, centre_x - radius_x)
        x1 = max_x if radius_x is None else min(max_x, centre_x + radius_x)
        y0 = -max_y if radius_y is None else max(-max_y, centre_y - radius_y)
        y1 = max_y if radius_y is None else min(max_y, centre_y + radius_y)
        if axis == "x":
            y0 = y1 = 0
        elif axis == "y":
            x0 = x1 = 0
        candidates = [
            (dx, dy)
            for dy in range(y0, y1 + 1, max(1, step))
            for dx in range(x0, x1 + 1, max(1, step))
        ]
    return _score_candidates_gpu(
        a_img,
        b_img,
        candidates,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
    )


def find_best_translation_gpu(
    a_img: Image.Image,
    b_img: Image.Image,
    *,
    axis: str,
    max_shift_x: int,
    max_shift_y: int,
    sigma: float,
    alpha_threshold: int,
    proxy_max_side: int,
):
    from anim_align_webp import _resize_proxy

    if axis == "none":
        return _search_gpu(
            a_img,
            b_img,
            axis="none",
            max_x=0,
            max_y=0,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
        )

    a_np = np.asarray(a_img.convert("RGBA"), dtype=np.uint8)
    b_np = np.asarray(b_img.convert("RGBA"), dtype=np.uint8)
    pa_np, sa = _resize_proxy(a_np, proxy_max_side)
    pb_np, sb = _resize_proxy(b_np, proxy_max_side)
    scale = min(sa, sb)

    if abs(sa - scale) > 1e-9:
        pa = a_img.resize(
            (max(1, round(a_img.width * scale)), max(1, round(a_img.height * scale))),
            Image.Resampling.BILINEAR,
        )
    else:
        pa = Image.fromarray(pa_np, "RGBA")
    if abs(sb - scale) > 1e-9:
        pb = b_img.resize(
            (max(1, round(b_img.width * scale)), max(1, round(b_img.height * scale))),
            Image.Resampling.BILINEAR,
        )
    else:
        pb = Image.fromarray(pb_np, "RGBA")

    pmx = max(0, round(max_shift_x * scale))
    pmy = max(0, round(max_shift_y * scale))
    largest = max(pmx if axis in ("x", "xy") else 0, pmy if axis in ("y", "xy") else 0)
    coarse_step = max(1, math.ceil(largest / 40))

    coarse = _search_gpu(
        pa,
        pb,
        axis=axis,
        max_x=pmx,
        max_y=pmy,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
        step=coarse_step,
    )
    if coarse_step > 1:
        coarse = _search_gpu(
            pa,
            pb,
            axis=axis,
            max_x=pmx,
            max_y=pmy,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
            centre_x=coarse.dx,
            centre_y=coarse.dy,
            radius_x=coarse_step,
            radius_y=coarse_step,
            step=1,
        )

    guess_x = round(coarse.dx / scale) if scale else 0
    guess_y = round(coarse.dy / scale) if scale else 0
    radius = max(3, math.ceil(1 / max(scale, 1e-9)))
    # The full-resolution refinement is already a tiny candidate set. CUDA evaluates
    # those exact scores in chunks, preserving the legacy integer-pixel answer.
    return _search_gpu(
        a_img,
        b_img,
        axis=axis,
        max_x=max_shift_x,
        max_y=max_shift_y,
        sigma=sigma,
        alpha_threshold=alpha_threshold,
        centre_x=guess_x,
        centre_y=guess_y,
        radius_x=radius,
        radius_y=radius,
        step=1,
    )


def find_best_scale_and_translation_gpu(
    anchor: Image.Image,
    image: Image.Image,
    *,
    axis: str,
    max_shift: int,
    sigma: float,
    alpha_threshold: int,
):
    """CUDA-backed equivalent of frame1_optimizer.find_best_scale_and_translation."""
    from frame1_optimizer import (
        _local_proxy_match,
        _proxy_pair,
        _scale_search_bounds,
        _to_source_shift,
        _uniform_resize,
    )

    if axis == "none":
        max_shift = 0

    low, high, guesses = _scale_search_bounds(anchor, image)
    coarse_count = 11
    coarse_step = (high - low) / max(1, coarse_count - 1)
    candidates = {low + coarse_step * i for i in range(coarse_count)}
    candidates.update(max(low, min(high, guess)) for guess in guesses)
    best: tuple[float, float, float, int, int] | None = None

    def consider(scale_value: float, max_side: int):
        nonlocal best
        pa, pb, world_scale = _proxy_pair(
            anchor,
            image,
            object_scale=scale_value,
            max_side=max_side,
            max_object_scale=high,
        )
        proxy_shift = max(0, round(max_shift * world_scale))
        step = max(1, math.ceil(proxy_shift / 10)) if axis != "none" else 1
        shift = _search_gpu(
            pa,
            pb,
            axis=axis,
            max_x=proxy_shift,
            max_y=proxy_shift,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
            step=step,
        )
        if step > 1 and axis != "none":
            shift = _search_gpu(
                pa,
                pb,
                axis=axis,
                max_x=proxy_shift,
                max_y=proxy_shift,
                sigma=sigma,
                alpha_threshold=alpha_threshold,
                centre_x=shift.dx,
                centre_y=shift.dy,
                radius_x=step,
                radius_y=step,
            )
        dx, dy = _to_source_shift(shift, world_scale, max_shift, axis)
        candidate = (shift.score, -abs(scale_value - 1.0), scale_value, dx, dy)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    for scale_value in sorted(candidates):
        consider(scale_value, 96)

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

    # Build the same two local proxy levels as the current CPU optimiser, but score
    # every local XY candidate on CUDA.
    for max_side, radius in ((384, 4), (1280, 3)):
        pa, pb, world_scale = _proxy_pair(
            anchor,
            image,
            object_scale=final_scale,
            max_side=max_side,
            max_object_scale=final_scale,
        )
        proxy_shift = max(0, round(max_shift * world_scale))
        centre_x = round(guess_dx * world_scale)
        centre_y = round(guess_dy * world_scale)
        result = _search_gpu(
            pa,
            pb,
            axis=axis,
            max_x=proxy_shift,
            max_y=proxy_shift,
            sigma=sigma,
            alpha_threshold=alpha_threshold,
            centre_x=centre_x,
            centre_y=centre_y,
            radius_x=radius,
            radius_y=radius,
        )
        guess_dx, guess_dy = _to_source_shift(result, world_scale, max_shift, axis)
        final_result = result

    from anim_align_webp import ShiftResult

    final_image = _uniform_resize(image, final_scale)
    return final_image, final_scale, ShiftResult(guess_dx, guess_dy, final_result.score)


def install(legacy_module) -> bool:
    """Patch the existing geometry pipeline in-place when CUDA is available.

    Any CUDA failure is handled by wrappers that immediately fall back to the
    existing CPU implementations, so enabling acceleration cannot make a job less
    robust on low-memory cards.
    """
    if not cuda_available():
        return False

    import anim_align_webp
    import frame1_optimizer

    cpu_translation = anim_align_webp.find_best_translation
    cpu_scale = frame1_optimizer.find_best_scale_and_translation

    def translation_wrapper(*args, **kwargs):
        try:
            return find_best_translation_gpu(*args, **kwargs)
        except Exception:
            return cpu_translation(*args, **kwargs)

    def scale_wrapper(*args, **kwargs):
        try:
            return find_best_scale_and_translation_gpu(*args, **kwargs)
        except Exception:
            return cpu_scale(*args, **kwargs)

    anim_align_webp.find_best_translation = translation_wrapper
    legacy.find_best_translation = translation_wrapper
    frame1_optimizer.find_best_scale_and_translation = scale_wrapper
    legacy.find_best_scale_and_translation = scale_wrapper
    return True
