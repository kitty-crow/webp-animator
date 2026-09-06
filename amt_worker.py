#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Match the allocator setting that made RIFE substantially less fragile on the
# 4 GB card. This is intentionally set before AMT imports torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from worker_common import (
    adaptive_midpoint_fill,
    content_bbox,
    is_cuda_oom,
    load_rgba,
    read_manifest,
    recursive_midpoints,
    release_cuda,
    write_result,
)

AMT_SCALES = (1.0, 0.75, 0.5, 0.25)
TILE_CORE = 256
TILE_CONTEXT = 64


def parse_args():
    parser = argparse.ArgumentParser(description="AMT selective interpolation worker")
    parser.add_argument("--amt-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def load_amt(source: Path, config: Path, checkpoint: Path):
    os.chdir(source)
    sys.path.insert(0, str(source))

    import torch
    from omegaconf import OmegaConf
    from utils.build_utils import build_from_cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network_cfg = OmegaConf.load(str(config)).network
    model = build_from_cfg(network_cfg)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device).eval()
    del ckpt, state

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        release_cuda(torch)
    return torch, device, model


def _tensor(torch, image: Image.Image, device, *, alpha: bool):
    if alpha:
        plane = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
        array = np.repeat(plane[..., None], 3, axis=2)
    else:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _run_model(torch, model, first_t, second_t, scale_factor: float, ratio: float):
    from utils.utils import InputPadder

    divisor = max(16, int(round(16 / max(scale_factor, 0.125))))
    padder = InputPadder(first_t.shape, divisor)
    first_p, second_p = padder.pad(first_t, second_t)
    embt = torch.tensor(float(ratio), device=first_t.device, dtype=first_t.dtype).view(1, 1, 1, 1)
    with torch.inference_mode():
        pred = model(first_p, second_p, embt, scale_factor=scale_factor, eval=True)["imgt_pred"]
    return padder.unpad(pred).clamp(0.0, 1.0)


def _infer_region(
    torch,
    device,
    model,
    first: Image.Image,
    second: Image.Image,
    *,
    alpha: bool,
    ratio: float,
):
    """Infer one region, progressively lowering AMT internal scale on CUDA OOM."""
    first_t = None
    second_t = None
    try:
        first_t = _tensor(torch, first, device, alpha=alpha)
        second_t = _tensor(torch, second, device, alpha=alpha)
        scales = AMT_SCALES if device.type == "cuda" else (1.0,)
        last_message = None

        for index, scale in enumerate(scales):
            pred = None
            try:
                pred = _run_model(torch, model, first_t, second_t, scale, ratio)
                if alpha:
                    result = pred[0, 0].mul(255).byte().cpu().numpy()
                else:
                    result = pred[0].mul(255).byte().cpu().numpy().transpose(1, 2, 0)
                del pred
                release_cuda(torch)
                return result, scale
            except Exception as exc:
                if not is_cuda_oom(torch, exc):
                    raise
                last_message = str(exc)
                del pred
                release_cuda(torch)
                if index + 1 < len(scales):
                    channel = "alpha" if alpha else "RGB"
                    print(
                        f"AMT CUDA OOM for {channel} at scale={scale:g}; "
                        f"retrying at scale={scales[index + 1]:g}",
                        file=sys.stderr,
                        flush=True,
                    )

        raise RuntimeError(last_message or "AMT inference exhausted its CUDA scale fallbacks")
    finally:
        del second_t, first_t
        release_cuda(torch)


def _interpolate_region(
    torch,
    device,
    model,
    first: Image.Image,
    second: Image.Image,
    ratio: float,
) -> Image.Image:
    rgb, rgb_scale = _infer_region(
        torch, device, model, first, second, alpha=False, ratio=ratio
    )

    a0 = np.asarray(first.getchannel("A"), dtype=np.uint8)
    a1 = np.asarray(second.getchannel("A"), dtype=np.uint8)
    if np.all(a0 == 255) and np.all(a1 == 255):
        alpha = np.full(a0.shape, 255, dtype=np.uint8)
        alpha_scale = rgb_scale
    else:
        release_cuda(torch)
        alpha, alpha_scale = _infer_region(
            torch, device, model, first, second, alpha=True, ratio=ratio
        )

    if rgb_scale != 1.0 or alpha_scale != 1.0:
        print(
            f"AMT_SCALE rgb={rgb_scale:g} alpha={alpha_scale:g}",
            file=sys.stderr,
            flush=True,
        )
    return Image.fromarray(np.dstack((rgb, alpha)), "RGBA")


def _interpolate_tiled(
    torch,
    device,
    model,
    first: Image.Image,
    second: Image.Image,
    ratio: float,
) -> Image.Image:
    """RIFE-style contextual tiling fallback for frames that still exceed VRAM."""
    width, height = first.size
    output = Image.new("RGBA", first.size, (0, 0, 0, 0))
    tile_count_x = (width + TILE_CORE - 1) // TILE_CORE
    tile_count_y = (height + TILE_CORE - 1) // TILE_CORE
    print(
        f"AMT_TILE_FALLBACK size={width}x{height} grid={tile_count_x}x{tile_count_y}",
        file=sys.stderr,
        flush=True,
    )

    for y0 in range(0, height, TILE_CORE):
        y1 = min(height, y0 + TILE_CORE)
        for x0 in range(0, width, TILE_CORE):
            x1 = min(width, x0 + TILE_CORE)
            ex0 = max(0, x0 - TILE_CONTEXT)
            ey0 = max(0, y0 - TILE_CONTEXT)
            ex1 = min(width, x1 + TILE_CONTEXT)
            ey1 = min(height, y1 + TILE_CONTEXT)

            tile0 = first.crop((ex0, ey0, ex1, ey1))
            tile1 = second.crop((ex0, ey0, ex1, ey1))
            tile_mid = _interpolate_region(torch, device, model, tile0, tile1, ratio)
            core = tile_mid.crop((x0 - ex0, y0 - ey0, x1 - ex0, y1 - ey0))
            output.paste(core, (x0, y0))
            del core, tile_mid, tile1, tile0
            release_cuda(torch)

    return output


def make_interpolator(torch, device, model):
    def interpolate(first: Image.Image, second: Image.Image, ratio: float = 0.5):
        if first.size != second.size:
            raise ValueError("AMT frames must share one canvas")
        ratio = max(0.0, min(1.0, float(ratio)))
        bbox = content_bbox(first, second)
        if bbox is None:
            return Image.new("RGBA", first.size, (0, 0, 0, 0))

        crop0 = first.crop(bbox)
        crop1 = second.crop(bbox)
        try:
            middle = _interpolate_region(torch, device, model, crop0, crop1, ratio)
        except Exception as exc:
            if not is_cuda_oom(torch, exc):
                raise
            release_cuda(torch)
            middle = _interpolate_tiled(torch, device, model, crop0, crop1, ratio)

        if crop0.size == first.size:
            return middle

        output = Image.new("RGBA", first.size, (0, 0, 0, 0))
        output.paste(middle, (bbox[0], bbox[1]))
        return output

    return interpolate


def _expected(task: dict) -> int:
    if "count" in task:
        count = max(0, int(task.get("count", 0)))
        return count if count > 0 else max(1, int(task.get("max_frames", 31)))
    return max(0, (2 ** int(task.get("depth", 1))) - 1)


def main():
    args = parse_args()
    manifest = read_manifest(args.manifest)
    output_dir = Path(manifest["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device, model = load_amt(args.amt_dir.resolve(), args.config.resolve(), args.checkpoint.resolve())
    interpolate = make_interpolator(torch, device, model)

    tasks = list(manifest.get("tasks", []))
    completed = 0
    total_expected = sum(_expected(task) for task in tasks)
    result_tasks = []

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
        if "count" in task:
            count = max(0, int(task.get("count", 0)))
            if count > 0:
                frames = []
                denominator = count + 1
                for numerator in range(1, denominator):
                    t = numerator / denominator
                    frames.append(save(interpolate(first, second, t), t))
                diagnostics = {"satisfied": True, "limit_reached": False}
            else:
                def generate(a, b):
                    return interpolate(a, b, 0.5)

                outcome = adaptive_midpoint_fill(
                    first,
                    second,
                    threshold=float(task.get("threshold", 12.0)),
                    max_frames=max(1, int(task.get("max_frames", 31))),
                    generate_midpoint=generate,
                    save_midpoint=save,
                    alpha_threshold=int(task.get("alpha_threshold", 8)),
                )
                frames = outcome["frames"]
                diagnostics = {
                    "satisfied": bool(outcome["satisfied"]),
                    "limit_reached": bool(outcome["limit_reached"]),
                    "max_score": float(outcome["max_score"]),
                }
        else:
            threshold = task.get("threshold")
            frames = recursive_midpoints(
                first,
                second,
                depth=max(0, int(task.get("depth", 1))),
                threshold=None if threshold is None else float(threshold),
                generate_midpoint=lambda a, b: interpolate(a, b, 0.5),
                save_midpoint=save,
                alpha_threshold=int(task.get("alpha_threshold", 8)),
            )

        result_tasks.append({
            "id": task.get("id", str(task_index)),
            "frames": frames,
            **diagnostics,
        })
        release_cuda(torch)

    write_result(args.result, {"tasks": result_tasks})


if __name__ == "__main__":
    main()
