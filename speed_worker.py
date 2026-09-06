#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from worker_common import (
    adaptive_midpoint_fill,
    alpha_midpoint,
    compose_rgb_with_alpha,
    content_bbox,
    fixed_midpoint_fill,
    is_cuda_oom,
    load_rgba,
    read_manifest,
    release_cuda,
    write_result,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SPEED midpoint generation worker")
    parser.add_argument("--speed-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _precision_for_device(torch, device) -> str:
    explicit = os.environ.get("SPEED_PRECISION")
    if explicit in {"fp32", "fp16", "bf16"}:
        return explicit
    if device.type != "cuda":
        return "fp32"
    major, _ = torch.cuda.get_device_capability(device)
    return "bf16" if major >= 8 else "fp16"


def _resize_pair(first: Image.Image, second: Image.Image, scale: float):
    if scale >= 0.999:
        return first, second
    size = (
        max(32, int(round(first.width * scale / 16)) * 16),
        max(32, int(round(first.height * scale / 16)) * 16),
    )
    return (
        first.resize(size, Image.Resampling.LANCZOS),
        second.resize(size, Image.Resampling.LANCZOS),
    )


def _expected(task: dict) -> int:
    if "count" not in task:
        return 1
    count = max(0, int(task.get("count", 0)))
    return count if count > 0 else max(1, int(task.get("max_frames", 31)))


def main():
    args = parse_args()
    speed_dir = args.speed_dir.resolve()
    os.chdir(speed_dir)
    sys.path.insert(0, str(speed_dir))

    import torch
    import inference as speed_inference

    manifest = read_manifest(args.manifest)
    output_dir = Path(manifest["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = speed_inference.get_device(None)
    precision = _precision_for_device(torch, device)
    model = speed_inference.build_model(
        str(args.config.resolve()),
        str(args.checkpoint.resolve()),
        device,
        strict_load=False,
    )
    model.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        release_cuda(torch)

    torch.manual_seed(int(manifest.get("seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(manifest.get("seed", 0)))

    def generate_midpoint(first: Image.Image, second: Image.Image):
        if first.size != second.size:
            raise ValueError("SPEED frames must share one canvas")
        bbox = content_bbox(first, second)
        if bbox is None:
            return Image.new("RGBA", first.size, (0, 0, 0, 0))

        crop0 = first.crop(bbox)
        crop1 = second.crop(bbox)
        original_size = crop0.size
        last_message = None
        rgb_result = None
        for scale in ((1.0, 0.75, 0.5, 0.375) if device.type == "cuda" else (1.0,)):
            try:
                scaled0, scaled1 = _resize_pair(crop0, crop1, scale)
                arr0 = np.asarray(scaled0.convert("RGB"), dtype=np.uint8)
                arr1 = np.asarray(scaled1.convert("RGB"), dtype=np.uint8)
                pred = speed_inference.interpolate_batch(
                    model,
                    [arr0],
                    [arr1],
                    device,
                    precision=precision,
                )[0]
                rgb_result = Image.fromarray(pred, "RGB")
                if rgb_result.size != original_size:
                    rgb_result = rgb_result.resize(original_size, Image.Resampling.LANCZOS)
                break
            except Exception as exc:
                if not is_cuda_oom(torch, exc):
                    raise
                last_message = str(exc)
                release_cuda(torch)
        if rgb_result is None:
            raise RuntimeError(last_message or "SPEED failed to generate a frame")

        alpha = alpha_midpoint(crop0, crop1)
        return compose_rgb_with_alpha(rgb_result, alpha, first.size, bbox)

    tasks = list(manifest.get("tasks", []))
    total_expected = sum(_expected(task) for task in tasks)
    completed = 0
    results = []

    for task_index, task in enumerate(tasks):
        first = load_rgba(Path(task["left"]))
        second = load_rgba(Path(task["right"]))
        counter = 0

        def save(image, t):
            nonlocal counter, completed
            path = output_dir / f"{task_index:04d}_{counter:04d}.png"
            counter += 1
            image.save(path)
            completed += 1
            print(f"PROGRESS {completed} {max(1, total_expected)}", flush=True)
            return {"path": str(path), "t": float(t)}

        diagnostics = {}
        if "count" not in task:
            ref = save(generate_midpoint(first, second), 0.5)
            refs = [ref]
        else:
            count = max(0, int(task.get("count", 0)))
            if count > 0:
                refs = fixed_midpoint_fill(
                    first,
                    second,
                    count=count,
                    generate_midpoint=generate_midpoint,
                    save_midpoint=save,
                    alpha_threshold=int(task.get("alpha_threshold", 8)),
                )
                diagnostics = {"satisfied": True, "limit_reached": False}
            else:
                outcome = adaptive_midpoint_fill(
                    first,
                    second,
                    threshold=float(task.get("threshold", 12.0)),
                    max_frames=max(1, int(task.get("max_frames", 31))),
                    generate_midpoint=generate_midpoint,
                    save_midpoint=save,
                    alpha_threshold=int(task.get("alpha_threshold", 8)),
                )
                refs = outcome["frames"]
                diagnostics = {
                    "satisfied": bool(outcome["satisfied"]),
                    "limit_reached": bool(outcome["limit_reached"]),
                    "max_score": float(outcome["max_score"]),
                }

        results.append({
            "id": task.get("id", str(task_index)),
            "frame": refs[0]["path"] if refs else None,
            "frames": refs,
            **diagnostics,
        })
        release_cuda(torch)

    write_result(args.result, {"tasks": results, "precision": precision})


if __name__ == "__main__":
    main()
