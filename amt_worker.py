#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from worker_common import content_bbox, load_rgba, read_manifest, recursive_midpoints, write_result


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
    return torch, device, model


def _tensor(torch, image: Image.Image, device):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _alpha_tensor(torch, image: Image.Image, device):
    alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
    array = np.repeat(alpha[..., None], 3, axis=2)
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _run_model(torch, model, first_t, second_t, scale_factor: float):
    from utils.utils import InputPadder

    divisor = max(16, int(round(16 / max(scale_factor, 0.125))))
    padder = InputPadder(first_t.shape, divisor)
    first_p, second_p = padder.pad(first_t, second_t)
    embt = torch.tensor(0.5, device=first_t.device, dtype=first_t.dtype).view(1, 1, 1, 1)
    with torch.inference_mode():
        pred = model(first_p, second_p, embt, scale_factor=scale_factor, eval=True)["imgt_pred"]
    pred = padder.unpad(pred)
    return pred.clamp(0.0, 1.0)


def _infer_with_fallback(torch, model, first_t, second_t):
    scales = (1.0, 0.75, 0.5, 0.25) if first_t.device.type == "cuda" else (1.0,)
    last_error = None
    for scale in scales:
        try:
            return _run_model(torch, model, first_t, second_t, scale)
        except torch.OutOfMemoryError as exc:
            last_error = exc
            torch.cuda.empty_cache()
            gc.collect()
    if last_error is not None:
        raise last_error
    raise RuntimeError("AMT inference failed")


def make_interpolator(torch, device, model):
    def interpolate(first: Image.Image, second: Image.Image):
        if first.size != second.size:
            raise ValueError("AMT frames must share one canvas")
        bbox = content_bbox(first, second)
        if bbox is None:
            return Image.new("RGBA", first.size, (0, 0, 0, 0))

        crop0 = first.crop(bbox)
        crop1 = second.crop(bbox)
        t0 = _tensor(torch, crop0, device)
        t1 = _tensor(torch, crop1, device)
        rgb_t = _infer_with_fallback(torch, model, t0, t1)
        rgb = rgb_t[0].mul(255).byte().cpu().numpy().transpose(1, 2, 0)

        a0 = np.asarray(crop0.getchannel("A"), dtype=np.uint8)
        a1 = np.asarray(crop1.getchannel("A"), dtype=np.uint8)
        if np.all(a0 == 255) and np.all(a1 == 255):
            alpha = np.full(a0.shape, 255, dtype=np.uint8)
        else:
            at0 = _alpha_tensor(torch, crop0, device)
            at1 = _alpha_tensor(torch, crop1, device)
            alpha_t = _infer_with_fallback(torch, model, at0, at1)
            alpha = alpha_t[0, 0].mul(255).byte().cpu().numpy()

        rgba = Image.fromarray(np.dstack((rgb, alpha)), "RGBA")
        output = Image.new("RGBA", first.size, (0, 0, 0, 0))
        output.alpha_composite(rgba, (bbox[0], bbox[1]))
        del t1, t0, rgb_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return output

    return interpolate


def main():
    args = parse_args()
    manifest = read_manifest(args.manifest)
    output_dir = Path(manifest["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device, model = load_amt(args.amt_dir.resolve(), args.config.resolve(), args.checkpoint.resolve())
    interpolate = make_interpolator(torch, device, model)

    tasks = list(manifest.get("tasks", []))
    completed = 0
    total_expected = sum(max(0, (2 ** int(task.get("depth", 1))) - 1) for task in tasks)
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

        threshold = task.get("threshold")
        frames = recursive_midpoints(
            first,
            second,
            depth=max(0, int(task.get("depth", 1))),
            threshold=None if threshold is None else float(threshold),
            generate_midpoint=interpolate,
            save_midpoint=save,
            alpha_threshold=int(task.get("alpha_threshold", 8)),
        )
        result_tasks.append({"id": task.get("id", str(task_index)), "frames": frames})
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    write_result(args.result, {"tasks": result_tasks})


if __name__ == "__main__":
    main()
