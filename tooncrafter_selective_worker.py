#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

# ToonCrafter uses the same older PyTorch generation as MoG in its isolated env.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

from worker_common import content_bbox, gap_score, is_cuda_oom, load_rgba, read_manifest, release_cuda, write_result

TARGETS = ((512, 320), (448, 256), (384, 224), (320, 192), (256, 160), (224, 128))
PREP_UNITS = 24
DEFAULT_DDIM_STEPS = int(os.environ.get("TOONCRAFTER_DDIM_STEPS", "50"))
WORKER_REVISION = "tooncrafter-selective-offload-v1"


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


def parse_args():
    parser = argparse.ArgumentParser(description="ToonCrafter selective interpolation worker")
    parser.add_argument("--tooncrafter-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _prep_heartbeat(stop: threading.Event):
    value = 1
    while not stop.wait(3.0):
        if _PROGRESS is None:
            continue
        value = min(PREP_UNITS - 1, value + 1)
        _PROGRESS.set(value)


def _target_ladder(total_vram: int) -> list[tuple[int, int]]:
    gb = total_vram / 1024**3
    if gb <= 4.5:
        return [(256, 160), (224, 128)]
    if gb <= 8.0:
        return [(320, 192), (256, 160), (224, 128)]
    if gb <= 12.0:
        return [(448, 256), (384, 224), (320, 192), (256, 160)]
    return list(TARGETS)


def _load_checkpoint_mmap(torch, model, checkpoint: Path):
    kwargs = {"map_location": "cpu"}
    try:
        state = torch.load(str(checkpoint), mmap=True, **kwargs)
    except TypeError:
        state = torch.load(str(checkpoint), **kwargs)

    state_dict = state.get("state_dict", state)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        from collections import OrderedDict

        renamed = OrderedDict(state_dict)
        for key in list(renamed.keys()):
            if "framestride_embed" in key:
                renamed[key.replace("framestride_embed", "fps_embedding")] = renamed.pop(key)
        model.load_state_dict(renamed, strict=True)
    del state, state_dict
    gc.collect()
    return model


def _auxiliary_modules(model):
    names = ("first_stage_model", "cond_stage_model", "embedder", "image_proj_model")
    return {name: getattr(model, name, None) for name in names}


def _move_core(model, device):
    auxiliaries = _auxiliary_modules(model)
    for name, module in auxiliaries.items():
        if module is not None:
            setattr(model, name, None)
    try:
        model.to(device)
    finally:
        for name, module in auxiliaries.items():
            if module is not None:
                setattr(model, name, module)
    return model


def _stage(model, name: str, device):
    module = getattr(model, name, None)
    if module is None:
        raise RuntimeError(f"ToonCrafter model is missing {name}")
    module.to(device)
    return module


def _offload(model, name: str, torch):
    module = getattr(model, name, None)
    if module is not None:
        module.cpu()
    release_cuda(torch)


def load_model(source: Path, config_path: Path, checkpoint: Path):
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "lvdm"))

    import torch
    from omegaconf import OmegaConf
    from utils.utils import instantiate_from_config

    if not torch.cuda.is_available():
        raise RuntimeError("ToonCrafter requires CUDA in the released implementation.")

    stop = threading.Event()
    thread = threading.Thread(target=_prep_heartbeat, args=(stop,), daemon=True)
    thread.start()
    try:
        print("ToonCrafter loading model configuration and checkpoint", flush=True)
        config = OmegaConf.load(str(config_path))
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
        model = _load_checkpoint_mmap(torch, model, checkpoint)
        model = model.half().eval()
        model.perframe_ae = True
        for module in _auxiliary_modules(model).values():
            if module is not None:
                module.half().eval().cpu()
    finally:
        stop.set()
        thread.join(timeout=1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    if _PROGRESS is not None:
        _PROGRESS.set(PREP_UNITS)
    print(
        f"TOONCRAFTER_WORKER {WORKER_REVISION} device={props.name} "
        f"vram={props.total_memory / 1024**3:.1f}GB component-offload=on",
        flush=True,
    )
    return torch, device, model, int(props.total_memory)


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


def _restore_frame(frame: Image.Image, placement, original_size, alpha: Image.Image):
    x, y, width, height = placement
    frame = frame.crop((x, y, x + width, y + height)).resize(original_size, Image.Resampling.LANCZOS)
    rgba = frame.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _encode_endpoints(torch, model, device, first_t, second_t, frames: int):
    from einops import rearrange

    print("ToonCrafter preparing endpoint latents", flush=True)
    _stage(model, "first_stage_model", device)
    try:
        first_video = first_t.unsqueeze(0).unsqueeze(2).repeat(1, 1, frames // 2, 1, 1)
        second_video = second_t.unsqueeze(0).unsqueeze(2).repeat(1, 1, frames - frames // 2, 1, 1)
        videos = torch.cat([first_video, second_video], dim=2)
        b, c, t, h, w = videos.shape
        x = rearrange(videos, "b c t h w -> (b t) c h w")
        posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)
        hidden_first_last = []
        for hidden in hidden_states:
            hidden = rearrange(hidden, "(b t) c h w -> b c t h w", b=b, t=t)
            hidden_first_last.append(
                torch.cat([hidden[:, :, 0:1], hidden[:, :, -1:]], dim=2).detach().cpu()
            )
        z = model.get_first_stage_encoding(posterior).detach()
        z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t).cpu()
        del posterior, hidden_states, videos, x, first_video, second_video
        return z, hidden_first_last
    finally:
        _offload(model, "first_stage_model", torch)


def _conditioning(torch, model, device, first_t, z, frames: int):
    print("ToonCrafter preparing text/image conditioning", flush=True)
    _stage(model, "cond_stage_model", device)
    try:
        text_emb = model.get_learned_conditioning([""]).detach()
    finally:
        _offload(model, "cond_stage_model", torch)

    _stage(model, "embedder", device)
    _stage(model, "image_proj_model", device)
    try:
        cond_images = model.embedder(first_t.unsqueeze(0))
        img_emb = model.image_proj_model(cond_images)
        cross = torch.cat([text_emb, img_emb], dim=1)
    finally:
        _offload(model, "embedder", torch)
        _offload(model, "image_proj_model", torch)

    z = z.to(device)
    endpoints = torch.zeros_like(z)
    endpoints[:, :, :1] = z[:, :, :1]
    endpoints[:, :, -1:] = z[:, :, -1:]
    del z, text_emb, img_emb, cond_images
    fs = torch.tensor([24], dtype=torch.long, device=device)
    return {"c_crossattn": [cross], "fs": fs, "c_concat": [endpoints]}


def _decode(torch, model, device, samples, hidden_states):
    print("ToonCrafter decoding generated frames with VAE offload", flush=True)
    # The denoiser is the dominant resident allocation. Move it to host RAM before
    # staging the VAE so 10-12 GB cards do not need both at once.
    _move_core(model, torch.device("cpu"))
    release_cuda(torch)
    _stage(model, "first_stage_model", device)
    try:
        hs = [item.to(device) for item in hidden_states]
        samples = samples.to(device)
        decoded = model.decode_first_stage(samples, ref_context=hs).detach().float().cpu()
        del hs, samples
        return decoded
    finally:
        _offload(model, "first_stage_model", torch)


def _generate_clip(torch, device, model, first: Image.Image, second: Image.Image, target: tuple[int, int], *, ddim_steps: int):
    from lvdm.models.samplers.ddim import DDIMSampler

    rgb0, placement0 = _fit_rgb(first, target)
    rgb1, placement1 = _fit_rgb(second, target)
    if placement0 != placement1:
        raise RuntimeError("ToonCrafter endpoint preparation became inconsistent.")

    def tensor(image):
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array.transpose(2, 0, 1)).to(dtype=torch.float16, device=device)

    first_t = tensor(rgb0)
    second_t = tensor(rgb1)
    frames = int(model.temporal_length)
    z, hidden_states = _encode_endpoints(torch, model, device, first_t, second_t, frames)
    cond = _conditioning(torch, model, device, first_t, z, frames)

    print(f"ToonCrafter denoising {target[0]}x{target[1]} for {ddim_steps} steps", flush=True)
    _move_core(model, device)
    channels = model.model.diffusion_model.out_channels
    h, w = target[1] // 8, target[0] // 8
    noise_shape = [1, channels, frames, h, w]
    sampler = DDIMSampler(model)
    fs = cond.pop("fs")

    def callback(_index):
        if _PROGRESS is not None:
            _PROGRESS.add(1)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        samples, _ = sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=1,
            shape=noise_shape[1:],
            verbose=False,
            callback=callback,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            eta=1.0,
            temporal_length=frames,
            fs=fs,
            timestep_spacing="uniform" if w == 32 else "uniform_trailing",
            guidance_rescale=0.0 if w == 32 else 0.7,
            clean_cond=True,
        )

    del cond, fs, sampler, first_t, second_t
    decoded = _decode(torch, model, device, samples.detach().cpu(), hidden_states)
    del samples, hidden_states
    release_cuda(torch)

    clip = decoded[0]
    output = []
    total_frames = int(clip.shape[1])
    for index in range(1, total_frames - 1):
        t = index / max(1, total_frames - 1)
        array = (
            clip[:, index]
            .clamp(-1, 1)
            .add(1)
            .mul(127.5)
            .byte()
            .numpy()
            .transpose(1, 2, 0)
        )
        frame = Image.fromarray(array, "RGB")
        alpha = Image.blend(first.getchannel("A"), second.getchannel("A"), t)
        output.append((t, _restore_frame(frame, placement0, first.size, alpha)))
    del decoded, clip
    return output


def _select_fixed(candidates: list[tuple[float, Image.Image]], count: int):
    count = max(0, min(int(count), len(candidates)))
    if count <= 0:
        return []
    available = list(candidates)
    selected = []
    for ordinal in range(1, count + 1):
        target = ordinal / (count + 1)
        item = min(available, key=lambda pair: abs(pair[0] - target))
        available.remove(item)
        selected.append(item)
    selected.sort(key=lambda pair: pair[0])
    return selected


def _select_adaptive(first, second, candidates, *, threshold: float, max_frames: int, alpha_threshold: int):
    selected: list[tuple[float, Image.Image]] = []
    remaining = list(candidates)
    max_frames = min(max(0, int(max_frames)), len(remaining))

    def sequence():
        return [(0.0, first)] + sorted(selected, key=lambda item: item[0]) + [(1.0, second)]

    while len(selected) < max_frames:
        seq = sequence()
        gaps = [gap_score(seq[index][1], seq[index + 1][1], alpha_threshold) for index in range(len(seq) - 1)]
        worst_index = max(range(len(gaps)), key=lambda index: gaps[index])
        if gaps[worst_index] <= threshold:
            break
        left_t, right_t = seq[worst_index][0], seq[worst_index + 1][0]
        inside = [item for item in remaining if left_t < item[0] < right_t]
        if not inside:
            break
        midpoint = (left_t + right_t) * 0.5
        choice = min(inside, key=lambda item: abs(item[0] - midpoint))
        remaining.remove(choice)
        selected.append(choice)

    seq = sequence()
    max_score = max(
        (gap_score(seq[index][1], seq[index + 1][1], alpha_threshold) for index in range(len(seq) - 1)),
        default=0.0,
    )
    selected.sort(key=lambda item: item[0])
    return selected, max_score


def main():
    global _PROGRESS
    args = parse_args()
    manifest = read_manifest(args.manifest)
    tasks = list(manifest.get("tasks", []))
    _PROGRESS = ProgressState(PREP_UNITS + max(1, len(tasks)) * DEFAULT_DDIM_STEPS * 2)
    _PROGRESS.set(0)

    output_dir = Path(manifest["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, device, model, total_vram = load_model(
        args.tooncrafter_dir.resolve(),
        args.config.resolve(),
        args.checkpoint.resolve(),
    )

    targets = _target_ladder(total_vram)
    result_tasks = []
    for task_index, task in enumerate(tasks):
        first_full = load_rgba(Path(task["left"]))
        second_full = load_rgba(Path(task["right"]))
        if first_full.size != second_full.size:
            raise ValueError("ToonCrafter frames must share one canvas")

        bbox = content_bbox(first_full, second_full, margin=64)
        if bbox is None:
            candidates = []
        else:
            first = first_full.crop(bbox)
            second = second_full.crop(bbox)
            last_error = None
            candidates = []
            for target_index, target in enumerate(targets):
                try:
                    if target_index:
                        print(f"ToonCrafter low-VRAM retry at {target[0]}x{target[1]}", flush=True)
                    local = _generate_clip(
                        torch,
                        device,
                        model,
                        first,
                        second,
                        target,
                        ddim_steps=DEFAULT_DDIM_STEPS,
                    )
                    for t, local_frame in local:
                        if bbox == (0, 0, first_full.width, first_full.height):
                            candidates.append((t, local_frame))
                            continue
                        canvas = Image.new("RGBA", first_full.size, (0, 0, 0, 0))
                        canvas.alpha_composite(local_frame, (bbox[0], bbox[1]))
                        candidates.append((t, canvas))
                    break
                except Exception as exc:
                    if not is_cuda_oom(torch, exc):
                        raise
                    last_error = exc
                    print(f"ToonCrafter CUDA OOM at {target[0]}x{target[1]}; retrying smaller", flush=True)
                    try:
                        _move_core(model, torch.device("cpu"))
                    except Exception:
                        pass
                    gc.collect()
                    release_cuda(torch)
            else:
                raise RuntimeError(
                    "ToonCrafter exhausted all low-VRAM fallbacks. Its diffusion core is "
                    "large enough that a 4 GB GTX 1050 Ti is likely below the practical "
                    "minimum even when the frame resolution is reduced. The same worker "
                    "uses FP16 component offload and larger quality rungs on 11 GB cards."
                ) from last_error

        count = max(0, int(task.get("count", 0)))
        threshold = float(task.get("threshold", 12.0))
        alpha_threshold = int(task.get("alpha_threshold", 8))
        if count > 0:
            chosen = _select_fixed(candidates, count)
            requested_too_many = count > len(candidates)
            max_score = None
        else:
            chosen, max_score = _select_adaptive(
                first_full,
                second_full,
                candidates,
                threshold=threshold,
                max_frames=max(1, int(task.get("max_frames", 31))),
                alpha_threshold=alpha_threshold,
            )
            requested_too_many = False

        refs = []
        for frame_index, (t, image) in enumerate(chosen):
            path = output_dir / f"{task_index:04d}_{frame_index:04d}.png"
            image.save(path)
            refs.append({"path": str(path), "t": float(t)})

        limit_reached = requested_too_many or (
            count == 0 and max_score is not None and max_score > threshold and len(chosen) >= len(candidates)
        )
        result_tasks.append({
            "id": task.get("id", str(task_index)),
            "frames": refs,
            "satisfied": not limit_reached,
            "limit_reached": bool(limit_reached),
            **({"max_score": float(max_score)} if max_score is not None else {}),
        })

    _PROGRESS.set(_PROGRESS.total)
    write_result(args.result, {"tasks": result_tasks})


if __name__ == "__main__":
    main()
