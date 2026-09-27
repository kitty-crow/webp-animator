#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

# MoG's isolated environment currently uses PyTorch 2.1. Do not inherit allocator
# flags exported for newer RIFE/ResShift runtimes that that release cannot parse.
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

from worker_common import (
    content_bbox,
    gap_score,
    is_cuda_oom,
    load_rgba,
    read_manifest,
    release_cuda,
    write_result,
)

TARGETS = ((512, 320), (448, 256), (384, 224), (320, 192), (256, 160), (224, 128))
PREP_UNITS = 12
DEFAULT_DDIM_STEPS = int(os.environ.get("MOG_DDIM_STEPS", "50"))
WORKER_REVISION = "mog-selective-low-vram-v2"


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
    parser = argparse.ArgumentParser(description="Motion-Aware Generative VFI selective worker")
    parser.add_argument("--mog-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("ani", "real"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _prep_heartbeat(stop: threading.Event):
    value = 1
    while not stop.wait(4.0):
        if _PROGRESS is None:
            continue
        value = min(PREP_UNITS - 1, value + 1)
        _PROGRESS.set(value)


def _target_ladder(total_vram: int) -> list[tuple[int, int]]:
    gb = total_vram / 1024**3
    if gb <= 4.5:
        # Avoid wasting time on resolutions that are implausible on a 4 GB card.
        return [(256, 160), (224, 128)]
    if gb <= 8.0:
        return [(384, 224), (320, 192), (256, 160), (224, 128)]
    return list(TARGETS)


def _install_sampler_progress(inference_module):
    classes = []
    for value in (
        getattr(inference_module, "DDIMSampler", None),
        getattr(inference_module, "DDIMSampler_multicond", None),
    ):
        if value is not None and value not in classes:
            classes.append(value)

    for cls in classes:
        if getattr(cls, "__webp_progress_wrapped__", False):
            continue
        original = cls.sample

        def sample(self, *args, __original=original, **kwargs):
            prior = kwargs.get("callback")

            def callback(index):
                if _PROGRESS is not None:
                    _PROGRESS.add(1)
                if prior:
                    prior(index)

            kwargs["callback"] = callback
            return __original(self, *args, **kwargs)

        cls.sample = sample
        cls.__webp_progress_wrapped__ = True


def _load_checkpoint_mmap(torch, model, checkpoint: Path):
    kwargs = {"map_location": "cpu"}
    try:
        state = torch.load(str(checkpoint), mmap=True, **kwargs)
    except TypeError:
        state = torch.load(str(checkpoint), **kwargs)

    if "state_dict" in list(state.keys()):
        state_dict = state["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            from collections import OrderedDict

            renamed = OrderedDict(state_dict)
            for key in list(renamed.keys()):
                if "framestride_embed" in key:
                    renamed[key.replace("framestride_embed", "fps_embedding")] = renamed.pop(key)
            model.load_state_dict(renamed, strict=True)
    else:
        from collections import OrderedDict

        module = state["module"]
        renamed = OrderedDict((key[16:], value) for key, value in module.items())
        model.load_state_dict(renamed, strict=True)
    del state
    gc.collect()
    return model


def _ensure_flow_checkpoint(source: Path, supplied: Path) -> Path:
    expected = source / "emavfi" / "ckpt" / "ours_t.ckpt"
    expected.parent.mkdir(parents=True, exist_ok=True)
    if supplied.resolve() == expected.resolve():
        return expected
    if not expected.is_file() or expected.stat().st_size != supplied.stat().st_size:
        shutil.copy2(supplied, expected)
    return expected


def load_model(source: Path, config_path: Path, checkpoint: Path, flow_checkpoint: Path):
    os.chdir(source)
    sys.path.insert(0, str(source))

    # MoG constructs its EMA-VFI motion model as part of the parent model, so its
    # conventional checkpoint has to exist before instantiate_from_config runs.
    _ensure_flow_checkpoint(source, flow_checkpoint)

    import torch
    from omegaconf import OmegaConf
    from emavfi.Trainer import Model as VFIModel

    # Stop the EMA-VFI constructor from creating a transient CUDA copy. We stage it
    # explicitly for optical-flow calculation later and immediately return it to CPU.
    original_device = VFIModel.device
    VFIModel.device = lambda self: None
    try:
        import scripts.evaluation.inference as mog_inference
        from utils.utils import instantiate_from_config

        _install_sampler_progress(mog_inference)
        config = OmegaConf.load(str(config_path))
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
    finally:
        VFIModel.device = original_device

    stop = threading.Event()
    thread = threading.Thread(target=_prep_heartbeat, args=(stop,), daemon=True)
    thread.start()
    try:
        model = _load_checkpoint_mmap(torch, model, checkpoint)
        # Convert weights before they ever reach VRAM. Moving FP32 first and casting
        # afterwards creates a disastrous peak on 4-11 GB cards.
        model = model.half().eval()
    finally:
        stop.set()
        thread.join(timeout=1)

    if not torch.cuda.is_available():
        raise RuntimeError("MoG requires CUDA in the released inference implementation.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    # Detach the separate motion network while moving the much larger diffusion
    # model to CUDA. This avoids a second large transient allocation. It is restored
    # immediately afterwards and remains on CPU between flow calculations.
    vfi = getattr(model, "vfi", None)
    if vfi is not None:
        model.vfi = None
    try:
        model = model.to(device)
    finally:
        if vfi is not None:
            model.vfi = vfi
    model.perframe_ae = True

    if vfi is not None and hasattr(vfi, "net"):
        vfi.net = vfi.net.float().cpu().eval()

    original_cal_flow = mog_inference.cal_flow

    def staged_cal_flow(videos, vfi_model):
        if hasattr(vfi_model, "net"):
            vfi_model.net = vfi_model.net.float().to(videos.device).eval()
        try:
            motion = original_cal_flow(videos.float(), vfi_model)
        finally:
            if hasattr(vfi_model, "net"):
                vfi_model.net = vfi_model.net.cpu().eval()
            release_cuda(torch)
        return motion

    mog_inference.cal_flow = staged_cal_flow
    if _PROGRESS is not None:
        _PROGRESS.set(PREP_UNITS)
    print(
        f"MOG_WORKER {WORKER_REVISION} device={props.name} "
        f"vram={props.total_memory / 1024**3:.1f}GB",
        flush=True,
    )
    return torch, device, model, mog_inference, int(props.total_memory)


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


def _video_tensor(torch, first: Image.Image, second: Image.Image, device, frames=16):
    def tensor(image):
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array.transpose(2, 0, 1)).to(dtype=torch.float16)

    first_t = tensor(first).unsqueeze(1).repeat(1, frames // 2, 1, 1)
    second_t = tensor(second).unsqueeze(1).repeat(1, frames - frames // 2, 1, 1)
    return torch.cat([first_t, second_t], dim=1).unsqueeze(0).to(device)


def _restore_frame(frame, placement, original_size, alpha: Image.Image):
    x, y, width, height = placement
    frame = frame.crop((x, y, x + width, y + height)).resize(
        original_size,
        Image.Resampling.LANCZOS,
    )
    rgba = frame.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _generate_clip(
    torch,
    model,
    inference_module,
    first: Image.Image,
    second: Image.Image,
    target: tuple[int, int],
    *,
    ddim_steps: int,
):
    rgb0, placement0 = _fit_rgb(first, target)
    rgb1, placement1 = _fit_rgb(second, target)
    if placement0 != placement1:
        raise RuntimeError("MoG endpoint preparation became inconsistent.")

    device = next(model.parameters()).device
    videos = _video_tensor(torch, rgb0, rgb1, device, frames=16)
    h, w = target[1] // 8, target[0] // 8
    channels = model.model.diffusion_model.out_channels
    noise_shape = [1, channels, 16, h, w]

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        variants = inference_module.image_guided_synthesis(
            model,
            ["natural motion between the two endpoint frames"],
            videos,
            noise_shape,
            n_samples=1,
            ddim_steps=ddim_steps,
            ddim_eta=1.0,
            # text_input=False turns the prompt into an empty sequence. CFG at 1.0
            # avoids an otherwise redundant second denoiser pass.
            unconditional_guidance_scale=1.0,
            cfg_img=None,
            fs=24,
            text_input=False,
            multiple_cond_cfg=False,
            loop=False,
            interp=True,
            timestep_spacing="uniform_trailing",
            guidance_rescale=0.0,
        )

    clip = variants[0, 0].detach().float().cpu().clamp(-1, 1)
    del variants, videos
    release_cuda(torch)

    output = []
    total_frames = int(clip.shape[1])
    for index in range(1, total_frames - 1):
        t = index / max(1, total_frames - 1)
        array = (
            clip[:, index]
            .add(1)
            .mul(127.5)
            .byte()
            .numpy()
            .transpose(1, 2, 0)
        )
        frame = Image.fromarray(array, "RGB")
        alpha = Image.blend(first.getchannel("A"), second.getchannel("A"), t)
        output.append((t, _restore_frame(frame, placement0, first.size, alpha)))
    del clip
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


def _select_adaptive(
    first: Image.Image,
    second: Image.Image,
    candidates: list[tuple[float, Image.Image]],
    *,
    threshold: float,
    max_frames: int,
    alpha_threshold: int,
):
    selected: list[tuple[float, Image.Image]] = []
    remaining = list(candidates)
    max_frames = min(max(0, int(max_frames)), len(remaining))

    def sequence():
        return [(0.0, first)] + sorted(selected, key=lambda item: item[0]) + [(1.0, second)]

    while len(selected) < max_frames:
        seq = sequence()
        gaps = [
            gap_score(seq[index][1], seq[index + 1][1], alpha_threshold)
            for index in range(len(seq) - 1)
        ]
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
        (
            gap_score(seq[index][1], seq[index + 1][1], alpha_threshold)
            for index in range(len(seq) - 1)
        ),
        default=0.0,
    )
    selected.sort(key=lambda item: item[0])
    return selected, max_score


def main():
    global _PROGRESS
    args = parse_args()
    manifest = read_manifest(args.manifest)
    tasks = list(manifest.get("tasks", []))

    # Reserve two DDIM passes per task. Most jobs finish in one. A 4 GB card may
    # need the second low-resolution pass, and either way every denoising step visibly
    # advances the existing progress UI instead of appearing frozen for minutes.
    _PROGRESS = ProgressState(
        PREP_UNITS + max(1, len(tasks)) * DEFAULT_DDIM_STEPS * 2
    )
    _PROGRESS.set(0)

    output_dir = Path(manifest["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, _device, model, inference_module, total_vram = load_model(
        args.mog_dir.resolve(),
        args.config.resolve(),
        args.checkpoint.resolve(),
        args.flow_checkpoint.resolve(),
    )

    targets = _target_ladder(total_vram)
    result_tasks = []
    for task_index, task in enumerate(tasks):
        first_full = load_rgba(Path(task["left"]))
        second_full = load_rgba(Path(task["right"]))
        if first_full.size != second_full.size:
            raise ValueError("MoG frames must share one canvas")

        bbox = content_bbox(first_full, second_full, margin=64)
        if bbox is None:
            candidates = []
        else:
            first = first_full.crop(bbox)
            second = second_full.crop(bbox)
            last_error = None
            for target_index, target in enumerate(targets):
                try:
                    if target_index:
                        print(
                            f"MoG low-VRAM retry at {target[0]}x{target[1]}",
                            file=sys.stderr,
                            flush=True,
                        )
                    local = _generate_clip(
                        torch,
                        model,
                        inference_module,
                        first,
                        second,
                        target,
                        ddim_steps=DEFAULT_DDIM_STEPS,
                    )
                    candidates = []
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
                    gc.collect()
                    release_cuda(torch)
            else:
                raise RuntimeError(
                    "MoG exhausted all low-VRAM resolutions. Its released checkpoint is "
                    "large enough that a 4 GB GTX 1050 Ti can be below the model's minimum "
                    "practical VRAM even in FP16. The same worker automatically starts at "
                    "higher quality on an 11 GB RTX 2080 Ti."
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
            count == 0
            and max_score is not None
            and max_score > threshold
            and len(chosen) >= len(candidates)
        )
        result_tasks.append({
            "id": task.get("id", str(task_index)),
            "frames": refs,
            "satisfied": not limit_reached,
            "limit_reached": bool(limit_reached),
            **({"max_score": float(max_score)} if max_score is not None else {}),
        })
        release_cuda(torch)

    _PROGRESS.set(_PROGRESS.total)
    write_result(args.result, {"tasks": result_tasks})


if __name__ == "__main__":
    main()
