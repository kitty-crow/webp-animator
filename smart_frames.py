from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

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


def _rgba_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0


def _active_mask(a: np.ndarray, b: np.ndarray, alpha_threshold: int) -> np.ndarray:
    threshold = alpha_threshold / 255.0
    mask = (a[..., 3] > threshold) | (b[..., 3] > threshold)
    if not np.any(mask):
        mask = np.ones(a.shape[:2], dtype=bool)
    return mask


def _edge_plane(array: np.ndarray) -> np.ndarray:
    alpha = array[..., 3]
    rgb = array[..., :3] * alpha[..., None]
    lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:] = np.abs(lum[:, 1:] - lum[:, :-1])
    gy[1:, :] = np.abs(lum[1:, :] - lum[:-1, :])
    return np.maximum(gx, gy)


def _score_arrays(a: np.ndarray, b: np.ndarray, alpha_threshold: int = 8) -> float:
    if a.shape != b.shape:
        raise ValueError("Smart frame analysis requires a common canvas.")

    mask = _active_mask(a, b, alpha_threshold)
    per_pixel = np.mean(np.abs(a - b), axis=2)
    active = per_pixel[mask]
    mean_error = float(active.mean()) if active.size else 0.0

    if active.size:
        k = max(1, int(math.ceil(active.size * 0.10)))
        if k >= active.size:
            top_error = float(active.mean())
        else:
            top_error = float(np.partition(active, active.size - k)[-k:].mean())
    else:
        top_error = 0.0

    edge_a = _edge_plane(a)
    edge_b = _edge_plane(b)
    edge_error = float(np.abs(edge_a - edge_b)[mask].mean()) if np.any(mask) else 0.0

    # Mean change keeps the score stable, top-decile change catches small but
    # meaningful moving structures, and edge change prevents blurred/structural
    # differences from being treated as redundant.
    score = 0.50 * mean_error + 0.35 * top_error + 0.15 * edge_error
    return max(0.0, min(100.0, score * 100.0))


def _cpu_pair_scores(frames: Sequence[Image.Image], pairs: Sequence[tuple[int, int]], alpha_threshold: int):
    arrays = [_rgba_array(frame) for frame in frames]
    workers = max(1, min(len(pairs) or 1, int(os.environ.get("FRAME_ANALYSIS_THREADS", os.cpu_count() or 1))))

    def score(pair):
        left, right = pair
        return _score_arrays(arrays[left], arrays[right], alpha_threshold)

    if workers <= 1 or len(pairs) <= 1:
        return [score(pair) for pair in pairs]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="frame-score") as pool:
        return list(pool.map(score, pairs))


def _gpu_pair_scores(frames: Sequence[Image.Image], pairs: Sequence[tuple[int, int]], alpha_threshold: int):
    torch = _torch_cuda()
    if torch is None:
        return None
    if not pairs:
        return []

    height, width = frames[0].height, frames[0].width
    for frame in frames:
        if frame.size != (width, height):
            raise ValueError("Smart frame analysis requires a common canvas.")

    device = torch.device("cuda")
    threshold = alpha_threshold / 255.0
    # Keep temporary tensors comfortably below the VRAM needed by the VFI models.
    bytes_per_pair = max(1, height * width * 4 * 4 * 3)
    target_bytes = int(os.environ.get("FRAME_ANALYSIS_GPU_BYTES", 192 * 1024 * 1024))
    batch_size = max(1, min(32, target_bytes // bytes_per_pair))
    results: list[float] = []

    try:
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            a_np = np.stack([np.asarray(frames[i].convert("RGBA"), dtype=np.uint8) for i, _ in chunk])
            b_np = np.stack([np.asarray(frames[j].convert("RGBA"), dtype=np.uint8) for _, j in chunk])
            a = torch.from_numpy(a_np).to(device=device, dtype=torch.float32).div_(255.0)
            b = torch.from_numpy(b_np).to(device=device, dtype=torch.float32).div_(255.0)

            active_mask = (a[..., 3] > threshold) | (b[..., 3] > threshold)
            per_pixel = (a - b).abs().mean(dim=3)

            alpha_a = a[..., 3]
            alpha_b = b[..., 3]
            lum_a = (
                a[..., 0] * alpha_a * 0.2126
                + a[..., 1] * alpha_a * 0.7152
                + a[..., 2] * alpha_a * 0.0722
            )
            lum_b = (
                b[..., 0] * alpha_b * 0.2126
                + b[..., 1] * alpha_b * 0.7152
                + b[..., 2] * alpha_b * 0.0722
            )
            edge_a = torch.zeros_like(lum_a)
            edge_b = torch.zeros_like(lum_b)
            edge_a[:, :, 1:] = torch.maximum(
                edge_a[:, :, 1:], (lum_a[:, :, 1:] - lum_a[:, :, :-1]).abs()
            )
            edge_a[:, 1:, :] = torch.maximum(
                edge_a[:, 1:, :], (lum_a[:, 1:, :] - lum_a[:, :-1, :]).abs()
            )
            edge_b[:, :, 1:] = torch.maximum(
                edge_b[:, :, 1:], (lum_b[:, :, 1:] - lum_b[:, :, :-1]).abs()
            )
            edge_b[:, 1:, :] = torch.maximum(
                edge_b[:, 1:, :], (lum_b[:, 1:, :] - lum_b[:, :-1, :]).abs()
            )
            edge_diff = (edge_a - edge_b).abs()

            for row in range(len(chunk)):
                mask = active_mask[row]
                if not bool(mask.any()):
                    mask = torch.ones_like(mask, dtype=torch.bool)
                values = per_pixel[row][mask]
                mean_error = values.mean()
                k = max(1, int(math.ceil(values.numel() * 0.10)))
                top_error = torch.topk(values, min(k, values.numel()), largest=True).values.mean()
                edge_error = edge_diff[row][mask].mean()
                score = 100.0 * (0.50 * mean_error + 0.35 * top_error + 0.15 * edge_error)
                results.append(float(score.clamp_(0.0, 100.0).item()))

            del edge_diff, edge_b, edge_a, lum_b, lum_a, per_pixel, active_mask, b, a
            torch.cuda.empty_cache()
        return results
    except Exception:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def pair_scores(
    frames: Sequence[Image.Image],
    pairs: Sequence[tuple[int, int]],
    *,
    alpha_threshold: int = 8,
) -> list[float]:
    if not pairs:
        return []
    gpu = _gpu_pair_scores(frames, pairs, alpha_threshold)
    if gpu is not None:
        return gpu
    return _cpu_pair_scores(frames, pairs, alpha_threshold)


def gap_scores(frames: Sequence[Image.Image], *, alpha_threshold: int = 8) -> list[float]:
    return pair_scores(
        frames,
        [(index, index + 1) for index in range(max(0, len(frames) - 1))],
        alpha_threshold=alpha_threshold,
    )


def _reduction_score_cpu(args) -> float:
    a, b, c, alpha_threshold = args
    aa = _rgba_array(a)
    bb = _rgba_array(b)
    cc = _rgba_array(c)
    predicted = (aa + cc) * 0.5
    return _score_arrays(bb, predicted, alpha_threshold)


def reduction_scores(frames: Sequence[Image.Image], *, alpha_threshold: int = 8) -> list[float]:
    """Return error for each interior frame versus the temporal midpoint of its neighbours.

    Result position 0 corresponds to frame index 1. Lower scores mean the middle
    frame contributes less unique temporal information.
    """
    if len(frames) < 3:
        return []

    torch = _torch_cuda()
    if torch is not None:
        try:
            device = torch.device("cuda")
            height, width = frames[0].height, frames[0].width
            bytes_per_triplet = max(1, height * width * 4 * 4 * 4)
            target_bytes = int(os.environ.get("FRAME_ANALYSIS_GPU_BYTES", 192 * 1024 * 1024))
            batch_size = max(1, min(24, target_bytes // bytes_per_triplet))
            result: list[float] = []
            for start in range(1, len(frames) - 1, batch_size):
                indexes = list(range(start, min(len(frames) - 1, start + batch_size)))
                a_np = np.stack([np.asarray(frames[i - 1].convert("RGBA"), dtype=np.uint8) for i in indexes])
                b_np = np.stack([np.asarray(frames[i].convert("RGBA"), dtype=np.uint8) for i in indexes])
                c_np = np.stack([np.asarray(frames[i + 1].convert("RGBA"), dtype=np.uint8) for i in indexes])
                a = torch.from_numpy(a_np).to(device=device, dtype=torch.float32).div_(255.0)
                b = torch.from_numpy(b_np).to(device=device, dtype=torch.float32).div_(255.0)
                c = torch.from_numpy(c_np).to(device=device, dtype=torch.float32).div_(255.0)
                predicted = (a + c) * 0.5

                # Reuse the same weighted metric, implemented directly on the batch.
                threshold = alpha_threshold / 255.0
                mask = (b[..., 3] > threshold) | (predicted[..., 3] > threshold)
                diff = (b - predicted).abs().mean(dim=3)

                def edge_lum(x):
                    alpha = x[..., 3]
                    lum = x[..., 0] * alpha * 0.2126 + x[..., 1] * alpha * 0.7152 + x[..., 2] * alpha * 0.0722
                    edge = torch.zeros_like(lum)
                    edge[:, :, 1:] = torch.maximum(edge[:, :, 1:], (lum[:, :, 1:] - lum[:, :, :-1]).abs())
                    edge[:, 1:, :] = torch.maximum(edge[:, 1:, :], (lum[:, 1:, :] - lum[:, :-1, :]).abs())
                    return edge

                edge_diff = (edge_lum(b) - edge_lum(predicted)).abs()
                for row in range(len(indexes)):
                    m = mask[row]
                    if not bool(m.any()):
                        m = torch.ones_like(m, dtype=torch.bool)
                    values = diff[row][m]
                    mean_error = values.mean()
                    k = max(1, int(math.ceil(values.numel() * 0.10)))
                    top_error = torch.topk(values, min(k, values.numel())).values.mean()
                    edge_error = edge_diff[row][m].mean()
                    score = 100.0 * (0.50 * mean_error + 0.35 * top_error + 0.15 * edge_error)
                    result.append(float(score.clamp_(0.0, 100.0).item()))
                del edge_diff, diff, mask, predicted, c, b, a
                torch.cuda.empty_cache()
            return result
        except Exception:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    jobs = [
        (frames[index - 1], frames[index], frames[index + 1], alpha_threshold)
        for index in range(1, len(frames) - 1)
    ]
    workers = max(1, min(len(jobs), int(os.environ.get("FRAME_ANALYSIS_THREADS", os.cpu_count() or 1))))
    if workers <= 1:
        return [_reduction_score_cpu(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="frame-reduce") as pool:
        return list(pool.map(_reduction_score_cpu, jobs))


def non_adjacent_removals(
    scores: Sequence[float],
    threshold: float,
    *,
    protected_indexes: Iterable[int] = (),
) -> list[int]:
    """Choose one safe parallel removal round from interior-frame scores."""
    protected = set(protected_indexes)
    candidates = [
        (float(score), offset + 1)
        for offset, score in enumerate(scores)
        if float(score) <= threshold and (offset + 1) not in protected
    ]
    candidates.sort()
    chosen: list[int] = []
    occupied: set[int] = set()
    for _, index in candidates:
        if index - 1 in occupied or index in occupied or index + 1 in occupied:
            continue
        chosen.append(index)
        occupied.add(index)
    return sorted(chosen)


def analyse(
    frames: Sequence[Image.Image],
    *,
    reduction_threshold: float,
    missing_threshold: float,
    alpha_threshold: int = 8,
) -> dict:
    gaps = gap_scores(frames, alpha_threshold=alpha_threshold)
    reductions = reduction_scores(frames, alpha_threshold=alpha_threshold)
    return {
        "gpu": _torch_cuda() is not None,
        "gap_scores": [round(value, 4) for value in gaps],
        "missing_gaps": [index + 1 for index, value in enumerate(gaps) if value > missing_threshold],
        "reduction_scores": [round(value, 4) for value in reductions],
        "reduction_candidates": [
            index + 2 for index, value in enumerate(reductions) if value <= reduction_threshold
        ],
        "reduction_threshold": float(reduction_threshold),
        "missing_threshold": float(missing_threshold),
    }
