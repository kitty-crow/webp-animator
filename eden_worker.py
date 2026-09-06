#!/usr/bin/env python3
from __future__ import annotations

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
    import argparse

    parser = argparse.ArgumentParser(description="EDEN midpoint generation worker")
    parser.add_argument("--eden-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _resize_pair(first: Image.Image, second: Image.Image, scale: float):
    if scale >= 0.999:
        return first, second
    width = max(32, int(round(first.width * scale / 32)) * 32)
    height = max(32, int(round(first.height * scale / 32)) * 32)
    return (
        first.resize((width, height), Image.Resampling.LANCZOS),
        second.resize((width, height), Image.Resampling.LANCZOS),
    )


def _expected(task: dict) -> int:
    if "count" not in task:
        return 1
    count = max(0, int(task.get("count", 0)))
    return count if count > 0 else max(1, int(task.get("max_frames", 31)))


def main():
    args = parse_args()
    eden_dir = args.eden_dir.resolve()
    os.chdir(eden_dir)
    sys.path.insert(0, str(eden_dir))

    import torch
    import yaml
    from src.models import load_model
    from src.transport import Sampler, create_transport
    from src.utils import InputPadder

    manifest = read_manifest(args.manifest)
    output_dir = Path(manifest["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.unsafe_load(args.config.resolve().read_text(encoding="utf-8"))
    model_args = dict(config["model_args"])
    if bool(model_args.get("use_xformers", False)):
        try:
            import xformers  # noqa: F401
        except Exception:
            model_args["use_xformers"] = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(config["model_name"], **model_args)
    checkpoint = torch.load(str(args.checkpoint.resolve()), map_location="cpu")
    model.load_state_dict(checkpoint["eden"])
    del checkpoint
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        release_cuda(torch)

    transport = create_transport("Linear", "velocity")
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method="euler",
        num_steps=2,
        atol=1e-6,
        rtol=1e-3,
    )
    cos_sim_mean = float(config["cos_sim_mean"])
    cos_sim_std = float(config["cos_sim_std"])
    vae_scaler = float(config["vae_scaler"])
    vae_shift = float(config["vae_shift"])
    latent_dim = int(model_args["latent_dim"])

    seed = int(manifest.get("seed", 0))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def interpolate_rgb(first: Image.Image, second: Image.Image):
        array0 = np.asarray(first.convert("RGB"), dtype=np.float32) / 255.0
        array1 = np.asarray(second.convert("RGB"), dtype=np.float32) / 255.0
        frame0 = torch.from_numpy(array0.transpose(2, 0, 1)).unsqueeze(0).to(device)
        frame1 = torch.from_numpy(array1.transpose(2, 0, 1)).unsqueeze(0).to(device)
        h, w = frame0.shape[2:]
        padder = InputPadder([h, w])
        difference = (
            (
                torch.mean(torch.cosine_similarity(frame0, frame1), dim=[1, 2])
                - cos_sim_mean
            )
            / cos_sim_std
        ).unsqueeze(1).to(device)
        cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))
        new_h, new_w = cond_frames.shape[2:]
        noise = torch.randn(
            [1, new_h // 32 * new_w // 32, latent_dim],
            device=device,
            dtype=frame0.dtype,
        )
        with torch.inference_mode():
            samples = sample_fn(
                noise,
                model.denoise,
                cond_frames=cond_frames,
                difference=difference,
            )[-1]
            latents = samples / vae_scaler + vae_shift
            generated = model.decode(latents)
            generated = padder.unpad(generated.clamp(0.0, 1.0))
        rgb = generated[0].mul(255).byte().cpu().numpy().transpose(1, 2, 0)
        del generated, latents, samples, noise, cond_frames, difference, frame1, frame0
        release_cuda(torch)
        return Image.fromarray(rgb, "RGB")

    def generate_midpoint(first: Image.Image, second: Image.Image):
        if first.size != second.size:
            raise ValueError("EDEN frames must share one canvas")
        bbox = content_bbox(first, second)
        if bbox is None:
            return Image.new("RGBA", first.size, (0, 0, 0, 0))

        crop0 = first.crop(bbox)
        crop1 = second.crop(bbox)
        original_size = crop0.size
        rgb_result = None
        last_message = None
        for scale in ((1.0, 0.75, 0.5, 0.375) if device.type == "cuda" else (1.0,)):
            try:
                scaled0, scaled1 = _resize_pair(crop0, crop1, scale)
                rgb_result = interpolate_rgb(scaled0, scaled1)
                if rgb_result.size != original_size:
                    rgb_result = rgb_result.resize(original_size, Image.Resampling.LANCZOS)
                break
            except Exception as exc:
                if not is_cuda_oom(torch, exc):
                    raise
                last_message = str(exc)
                release_cuda(torch)
        if rgb_result is None:
            raise RuntimeError(last_message or "EDEN failed to generate a frame")
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

    write_result(args.result, {"tasks": results})


if __name__ == "__main__":
    main()
