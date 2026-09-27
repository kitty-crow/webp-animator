#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from worker_common import (
    adaptive_midpoint_fill,
    content_bbox,
    is_cuda_oom,
    load_rgba,
    read_manifest,
    release_cuda,
    write_result,
)

# The published demo uses 448x256. Lower rungs are intentionally conservative
# low-VRAM fallbacks for cards such as the GTX 1050 Ti.
TARGETS = ((448, 256), (384, 224), (320, 192), (256, 160), (224, 128))
PREP_UNITS = 8
DEFAULT_DIFFUSION_STEPS = 15
WORKER_REVISION = "resshift-selective-low-vram-v1"


class ProgressState:
    def __init__(self, total: int):
        self.total = max(1, int(total))
        self.current = 0
        self.lock = threading.Lock()

    def set(self, value: int):
        with self.lock:
            self.current = max(self.current, min(self.total, int(value)))
            print(f"PROGRESS {self.current} {self.total}", flush=True)

    def add(self, amount: int = 1):
        with self.lock:
            self.current = min(self.total, self.current + int(amount))
            print(f"PROGRESS {self.current} {self.total}", flush=True)


_PROGRESS: ProgressState | None = None


class ProgressTqdm:
    """tqdm-compatible hook used by the upstream reverse diffusion loop."""

    def __init__(self, total=None, desc=None, **_kwargs):
        self.total = int(total or DEFAULT_DIFFUSION_STEPS)
        self.desc = desc

    def update(self, amount=1):
        if _PROGRESS is not None:
            _PROGRESS.add(int(amount))

    def close(self):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Input ResShift selective VFI worker")
    parser.add_argument("--resshift-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _expected(task: dict) -> int:
    count = max(0, int(task.get("count", 0)))
    if count > 0:
        return count
    return max(1, int(task.get("max_frames", 31)))


def _prep_heartbeat(stop: threading.Event):
    value = 1
    while not stop.wait(3.0):
        if _PROGRESS is None:
            continue
        value = min(PREP_UNITS - 1, value + 1)
        _PROGRESS.set(value)


def load_model(source: Path, model_dir: Path):
    os.chdir(source)
    sys.path.insert(0, str(source))

    import torch
    import model.model as model_module
    from model.hub import MultiInputResShiftHub

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Multi-Input ResShift uses CUDA/CuPy warping and requires an NVIDIA CUDA GPU."
        )

    model_module.tqdm = ProgressTqdm
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(
        f"RESSHIFT_WORKER {WORKER_REVISION} device={props.name} "
        f"vram={props.total_memory / 1024**3:.1f}GB",
        flush=True,
    )

    stop = threading.Event()
    thread = threading.Thread(target=_prep_heartbeat, args=(stop,), daemon=True)
    thread.start()
    try:
        model = MultiInputResShiftHub.from_pretrained(str(model_dir))
        model = model.to(device).eval().requires_grad_(False)
    finally:
        stop.set()
        thread.join(timeout=1)

    if _PROGRESS is not None:
        _PROGRESS.set(PREP_UNITS)
    torch.backends.cudnn.benchmark = False
    release_cuda(torch)
    return torch, device, model


def _fit_rgb(image: Image.Image, target: tuple[int, int]):
    target_w, target_h = target
    width, height = image.size
    scale = min(target_w / max(1, width), target_h / max(1, height))
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    rgba = image.convert("RGBA").resize((resized_w, resized_h), Image.Resampling.LANCZOS)

    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32)[..., None] / 255.0
    rgb = np.asarray(rgba.convert("RGB"), dtype=np.float32)
    premultiplied = np.clip(np.rint(rgb * alpha), 0, 255).astype(np.uint8)
    canvas = Image.new("RGB", target, (0, 0, 0))
    x = (target_w - resized_w) // 2
    y = (target_h - resized_h) // 2
    canvas.paste(Image.fromarray(premultiplied, "RGB"), (x, y))
    return canvas, (x, y, resized_w, resized_h)


def _to_tensor(torch, image: Image.Image, device):
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _tensor_rgb(tensor) -> Image.Image:
    array = (
        tensor[0]
        .detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .byte()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    return Image.fromarray(array, "RGB")


def _alpha_at(first: Image.Image, second: Image.Image, ratio: float) -> Image.Image:
    return Image.blend(first.getchannel("A"), second.getchannel("A"), float(ratio))


def _restore(
    rgb: Image.Image,
    placement: tuple[int, int, int, int],
    original_size: tuple[int, int],
):
    x, y, width, height = placement
    crop = rgb.crop((x, y, x + width, y + height))
    return crop.resize(original_size, Image.Resampling.LANCZOS)


def _prepare_pair(torch, device, model, first: Image.Image, second: Image.Image, target):
    rgb0, placement0 = _fit_rgb(first, target)
    rgb1, placement1 = _fit_rgb(second, target)
    if placement0 != placement1:
        raise RuntimeError("ResShift endpoint preparation became inconsistent.")
    input0 = _to_tensor(torch, rgb0, device)
    input1 = _to_tensor(torch, rgb1, device)
    with torch.inference_mode():
        flows = model.flow_model(input0, input1)
    return input0, input1, flows, placement0


def _generate_prepared(
    torch,
    model,
    input0,
    input1,
    flows,
    placement,
    crop0: Image.Image,
    crop1: Image.Image,
    ratio: float,
):
    with torch.inference_mode():
        generated = model.reverse_process([input0, input1], float(ratio), flows=flows)
    rgb = _restore(_tensor_rgb(generated), placement, crop0.size)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(_alpha_at(crop0, crop1, ratio))
    del generated
    release_cuda(torch)
    return rgba


def _generate_ratios(
    torch,
    device,
    model,
    first: Image.Image,
    second: Image.Image,
    ratios: list[float],
):
    if first.size != second.size:
        raise ValueError("ResShift frames must share one canvas")
    bbox = content_bbox(first, second, margin=48)
    if bbox is None:
        return [Image.new("RGBA", first.size, (0, 0, 0, 0)) for _ in ratios]

    crop0 = first.crop(bbox)
    crop1 = second.crop(bbox)
    last_error = None

    for target_index, target in enumerate(TARGETS):
        input0 = input1 = flows = None
        try:
            if target_index:
                print(
                    f"ResShift low-VRAM retry at {target[0]}x{target[1]} internal resolution",
                    file=sys.stderr,
                    flush=True,
                )
            input0, input1, flows, placement = _prepare_pair(
                torch, device, model, crop0, crop1, target
            )
            middles = [
                _generate_prepared(
                    torch,
                    model,
                    input0,
                    input1,
                    flows,
                    placement,
                    crop0,
                    crop1,
                    ratio,
                )
                for ratio in ratios
            ]
            outputs = []
            for middle in middles:
                if crop0.size == first.size:
                    outputs.append(middle)
                    continue
                output = Image.new("RGBA", first.size, (0, 0, 0, 0))
                output.alpha_composite(middle, (bbox[0], bbox[1]))
                outputs.append(output)
            del flows, input1, input0
            release_cuda(torch)
            return outputs
        except Exception as exc:
            if not is_cuda_oom(torch, exc):
                raise
            last_error = exc
            del flows, input1, input0
            gc.collect()
            release_cuda(torch)

    raise RuntimeError(
        "ResShift exhausted all low-VRAM resolution fallbacks. Close other CUDA "
        "applications or use RIFE/AMT for this render."
    ) from last_error


def main():
    global _PROGRESS
    args = parse_args()
    manifest = read_manifest(args.manifest)
    tasks = list(manifest.get("tasks", []))
    total_expected = sum(_expected(task) for task in tasks)
    _PROGRESS = ProgressState(PREP_UNITS + max(1, total_expected) * DEFAULT_DIFFUSION_STEPS)
    _PROGRESS.set(0)

    output_dir = Path(manifest["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device, model = load_model(
        args.resshift_dir.resolve(),
        args.model_dir.resolve(),
    )

    # If the upstream checkpoint changes its diffusion step count, keep the public
    # progress bar monotonic. The final completion update closes any estimate gap.
    result_tasks = []
    for task_index, task in enumerate(tasks):
        first = load_rgba(Path(task["left"]))
        second = load_rgba(Path(task["right"]))
        count = max(0, int(task.get("count", 0)))
        threshold = float(task.get("threshold", 12.0))
        alpha_threshold = int(task.get("alpha_threshold", 8))
        counter = 0

        def save(image: Image.Image, t: float):
            nonlocal counter
            path = output_dir / f"{task_index:04d}_{counter:04d}.png"
            counter += 1
            image.save(path)
            return {"path": str(path), "t": float(t)}

        diagnostics = {"satisfied": True, "limit_reached": False}
        if count > 0:
            ratios = [index / (count + 1) for index in range(1, count + 1)]
            images = _generate_ratios(torch, device, model, first, second, ratios)
            refs = [save(image, ratio) for image, ratio in zip(images, ratios)]
        else:
            # Automatic fill is inherently recursive because each generated midpoint
            # becomes an endpoint for the next worst sub-gap.
            def generate(a: Image.Image, b: Image.Image):
                return _generate_ratios(torch, device, model, a, b, [0.5])[0]

            outcome = adaptive_midpoint_fill(
                first,
                second,
                threshold=threshold,
                max_frames=max(1, int(task.get("max_frames", 31))),
                generate_midpoint=generate,
                save_midpoint=save,
                alpha_threshold=alpha_threshold,
            )
            refs = outcome["frames"]
            diagnostics = {
                "satisfied": bool(outcome["satisfied"]),
                "limit_reached": bool(outcome["limit_reached"]),
                "max_score": float(outcome["max_score"]),
            }

        result_tasks.append({
            "id": task.get("id", str(task_index)),
            "frames": refs,
            **diagnostics,
        })
        release_cuda(torch)

    _PROGRESS.set(_PROGRESS.total)
    write_result(args.result, {"tasks": result_tasks})


if __name__ == "__main__":
    main()
