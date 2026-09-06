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
    """Return RGBA in 0..1 with RGB premultiplied by alpha.

    Hidden RGB underneath fully transparent pixels is not visible animation data
    and must never make a frame look different or structurally important.
    """
    array = np.asarray(image.convert("RGBA"), dtype=np.float32).copy() / 255.0
    array[..., :3] *= array[..., 3:4]
    return array


def _premultiply_torch(tensor):
    tensor = tensor.clone()
    tensor[..., :3] *= tensor[..., 3:4]
    return tensor


def _active_mask(a: np.ndarray, b: np.ndarray, alpha_threshold: int) -> np.ndarray:
    threshold = alpha_threshold / 255.0
    return (a[..., 3] > threshold) | (b[..., 3] > threshold)


def _edge_plane(array: np.ndarray) -> np.ndarray:
    rgb = array[..., :3]
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
    if not np.any(mask):
        return 0.0

    per_pixel = np.mean(np.abs(a - b), axis=2)
    active = per_pixel[mask]
    mean_error = float(active.mean())
    k = max(1, int(math.ceil(active.size * 0.10)))
    top_error = (
        float(active.mean())
        if k >= active.size
        else float(np.partition(active, active.size - k)[-k:].mean())
    )
    edge_error = float(np.abs(_edge_plane(a) - _edge_plane(b))[mask].mean())

    score = 0.50 * mean_error + 0.35 * top_error + 0.15 * edge_error
    return max(0.0, min(100.0, score * 100.0))


def _cpu_pair_scores(
    frames: Sequence[Image.Image],
    pairs: Sequence[tuple[int, int]],
    alpha_threshold: int,
):
    arrays = [_rgba_array(frame) for frame in frames]
    workers = max(
        1,
        min(
            len(pairs) or 1,
            int(os.environ.get("FRAME_ANALYSIS_THREADS", os.cpu_count() or 1)),
        ),
    )

    def score(pair):
        left, right = pair
        return _score_arrays(arrays[left], arrays[right], alpha_threshold)

    if workers <= 1 or len(pairs) <= 1:
        return [score(pair) for pair in pairs]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="frame-score") as pool:
        return list(pool.map(score, pairs))


def _torch_edge(tensor):
    lum = (
        tensor[..., 0] * 0.2126
        + tensor[..., 1] * 0.7152
        + tensor[..., 2] * 0.0722
    )
    edge = tensor.new_zeros(lum.shape)
    edge[:, :, 1:] = edge[:, :, 1:].maximum(
        (lum[:, :, 1:] - lum[:, :, :-1]).abs()
    )
    edge[:, 1:, :] = edge[:, 1:, :].maximum(
        (lum[:, 1:, :] - lum[:, :-1, :]).abs()
    )
    return edge


def _score_torch_batch(torch, a, b, alpha_threshold: int) -> list[float]:
    threshold = alpha_threshold / 255.0
    mask = (a[..., 3] > threshold) | (b[..., 3] > threshold)
    per_pixel = (a - b).abs().mean(dim=3)
    edge_diff = (_torch_edge(a) - _torch_edge(b)).abs()
    results = []

    for row in range(a.shape[0]):
        active = mask[row]
        if not bool(active.any()):
            results.append(0.0)
            continue
        values = per_pixel[row][active]
        mean_error = values.mean()
        k = max(1, int(math.ceil(values.numel() * 0.10)))
        top_error = torch.topk(values, min(k, values.numel()), largest=True).values.mean()
        edge_error = edge_diff[row][active].mean()
        score = 100.0 * (0.50 * mean_error + 0.35 * top_error + 0.15 * edge_error)
        results.append(float(score.clamp(0.0, 100.0).item()))
    return results


def _gpu_pair_scores(
    frames: Sequence[Image.Image],
    pairs: Sequence[tuple[int, int]],
    alpha_threshold: int,
):
    torch = _torch_cuda()
    if torch is None:
        return None
    if not pairs:
        return []

    height, width = frames[0].height, frames[0].width
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError("Smart frame analysis requires a common canvas.")

    device = torch.device("cuda")
    bytes_per_pair = max(1, height * width * 4 * 4 * 3)
    target_bytes = int(os.environ.get("FRAME_ANALYSIS_GPU_BYTES", 192 * 1024 * 1024))
    batch_size = max(1, min(32, target_bytes // bytes_per_pair))
    results: list[float] = []

    try:
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            a_np = np.stack([
                np.asarray(frames[i].convert("RGBA"), dtype=np.uint8)
                for i, _ in chunk
            ])
            b_np = np.stack([
                np.asarray(frames[j].convert("RGBA"), dtype=np.uint8)
                for _, j in chunk
            ])
            a = torch.from_numpy(a_np).to(device=device, dtype=torch.float32).div_(255.0)
            b = torch.from_numpy(b_np).to(device=device, dtype=torch.float32).div_(255.0)
            a = _premultiply_torch(a)
            b = _premultiply_torch(b)
            results.extend(_score_torch_batch(torch, a, b, alpha_threshold))
            del b, a
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


def reduction_scores(
    frames: Sequence[Image.Image],
    *,
    alpha_threshold: int = 8,
) -> list[float]:
    """Error of every interior frame versus the midpoint of its neighbours.

    Result position 0 corresponds to frame index 1. Lower scores mean the middle
    frame contributes less unique temporal information.
    """
    if len(frames) < 3:
        return []

    torch = _torch_cuda()
    if torch is not None:
        try:
            height, width = frames[0].height, frames[0].width
            if any(frame.size != (width, height) for frame in frames):
                raise ValueError("Smart frame analysis requires a common canvas.")
            device = torch.device("cuda")
            bytes_per_triplet = max(1, height * width * 4 * 4 * 4)
            target_bytes = int(os.environ.get("FRAME_ANALYSIS_GPU_BYTES", 192 * 1024 * 1024))
            batch_size = max(1, min(24, target_bytes // bytes_per_triplet))
            results: list[float] = []

            for start in range(1, len(frames) - 1, batch_size):
                indexes = list(range(start, min(len(frames) - 1, start + batch_size)))
                a_np = np.stack([
                    np.asarray(frames[i - 1].convert("RGBA"), dtype=np.uint8)
                    for i in indexes
                ])
                b_np = np.stack([
                    np.asarray(frames[i].convert("RGBA"), dtype=np.uint8)
                    for i in indexes
                ])
                c_np = np.stack([
                    np.asarray(frames[i + 1].convert("RGBA"), dtype=np.uint8)
                    for i in indexes
                ])
                a = _premultiply_torch(
                    torch.from_numpy(a_np).to(device=device, dtype=torch.float32).div_(255.0)
                )
                b = _premultiply_torch(
                    torch.from_numpy(b_np).to(device=device, dtype=torch.float32).div_(255.0)
                )
                c = _premultiply_torch(
                    torch.from_numpy(c_np).to(device=device, dtype=torch.float32).div_(255.0)
                )
                predicted = (a + c) * 0.5
                results.extend(_score_torch_batch(torch, b, predicted, alpha_threshold))
                del predicted, c, b, a
                torch.cuda.empty_cache()
            return results
        except Exception:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    jobs = [
        (frames[index - 1], frames[index], frames[index + 1], alpha_threshold)
        for index in range(1, len(frames) - 1)
    ]
    workers = max(
        1,
        min(
            len(jobs),
            int(os.environ.get("FRAME_ANALYSIS_THREADS", os.cpu_count() or 1)),
        ),
    )
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
