#!/usr/bin/env python3
from __future__ import annotations

import shutil

import mog_selective_worker_v2 as implementation
from model_offload import apply_model_offload, model_nbytes


# The released MoG checkpoints are roughly ten gigabytes in FP32. Convert the
# freshly instantiated parameter storage to FP16 before load_state_dict copies the
# mmap-backed checkpoint into it. This avoids holding a full-size FP32 model plus a
# full-size checkpoint resident in host RAM on 16 GB machines.
_original_load_checkpoint = implementation._load_checkpoint_mmap


def _load_checkpoint_low_ram(torch, model, checkpoint):
    model = model.half().eval()
    return _original_load_checkpoint(torch, model, checkpoint)


implementation._load_checkpoint_mmap = _load_checkpoint_low_ram


# MoG is inconsistent about this filename: the published folder contains
# ours_t.pkl, older integration paths refer to ours_t.ckpt, and upstream Trainer.py
# loads ours_t.pkl directly. Keep both names present even when MOG_FLOW_CHECKPOINT
# points at a custom copy so model construction cannot fail on the alternate name.
def _ensure_flow_checkpoint(source, supplied):
    checkpoint_dir = source / "emavfi" / "ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pkl = checkpoint_dir / "ours_t.pkl"
    ckpt = checkpoint_dir / "ours_t.ckpt"
    supplied = supplied.resolve()

    for target in (pkl, ckpt):
        try:
            same = supplied == target.resolve()
        except OSError:
            same = False
        if same:
            continue
        if not target.is_file() or target.stat().st_size != supplied.stat().st_size:
            shutil.copy2(supplied, target)
    return ckpt


implementation._ensure_flow_checkpoint = _ensure_flow_checkpoint


# Upstream EMA-VFI calls torch.load() without map_location while constructing its
# motion model. Its published checkpoint contains CUDA storage tags, so that call
# otherwise deserialises directly onto GPU before our staged/offload logic can run.
# Force only unspecified torch.load calls to CPU during MoG model construction.
#
# More importantly, intercept the single top-level model.to(cuda) performed by the
# legacy worker. If the full FP16 diffusion model cannot coexist with activation
# headroom on the GPU, model_offload streams layers from RAM, or memory-maps them
# from disk if physical RAM is tight. This is intentionally allowed to be slow: a
# 4 GB card should degrade to transfer bandwidth, not fail merely because the whole
# checkpoint cannot reside in VRAM at once.
_original_load_model = implementation.load_model


def _load_model_offloaded(source, config_path, checkpoint, flow_checkpoint):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MoG requires a working CUDA runtime.")

    original_torch_load = torch.load
    original_module_to = torch.nn.Module.to
    routed = {"done": False}

    def cpu_default_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("map_location", "cpu")
        return original_torch_load(*load_args, **load_kwargs)

    def routed_module_to(module, *to_args, **to_kwargs):
        # Lightning's own .to() has already updated model.device before reaching
        # torch.nn.Module.to, which is useful because upstream allocates sampling
        # tensors via model.device even when Accelerate keeps parameters on meta/CPU.
        if not routed["done"]:
            try:
                size = model_nbytes(module)
            except Exception:
                size = 0
            # The top-level MoG diffusion model is multi-GB. Ignore ordinary child
            # module transfers (notably the separately staged optical-flow network).
            if size >= 512 * 1024**2:
                routed["done"] = True
                torch.nn.Module.to = original_module_to
                try:
                    placed, mode, _stats = apply_model_offload(
                        module,
                        torch.device("cuda"),
                        source=source,
                        checkpoint=checkpoint,
                        env_var="MOG_OFFLOAD",
                        namespace="mog",
                    )
                    return placed
                finally:
                    torch.nn.Module.to = routed_module_to
        return original_module_to(module, *to_args, **to_kwargs)

    torch.load = cpu_default_load
    torch.nn.Module.to = routed_module_to
    try:
        model_tuple = _original_load_model(source, config_path, checkpoint, flow_checkpoint)
        if not routed["done"]:
            raise RuntimeError("MoG model placement hook did not run; refusing an unqualified load path.")
        return model_tuple
    finally:
        torch.load = original_torch_load
        torch.nn.Module.to = original_module_to


implementation.load_model = _load_model_offloaded


# Accelerate CPU/disk offload leaves actual parameters on CPU/meta between calls.
# The original helper inferred the input device from next(model.parameters()), which
# is therefore no longer reliable. Lightning's model.device remains the intended
# CUDA execution device; model_offload also stores it explicitly.
_original_generate_clip = implementation._generate_clip


def _generate_clip_offloaded(
    torch,
    model,
    inference_module,
    first,
    second,
    target,
    *,
    ddim_steps,
):
    rgb0, placement0 = implementation._fit_rgb(first, target)
    rgb1, placement1 = implementation._fit_rgb(second, target)
    if placement0 != placement1:
        raise RuntimeError("MoG endpoint preparation became inconsistent.")

    device = getattr(model, "_webp_execution_device", None) or getattr(model, "device", None)
    if device is None or str(device).startswith("meta"):
        device = torch.device("cuda")
    videos = implementation._video_tensor(torch, rgb0, rgb1, device, frames=16)
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
    implementation.release_cuda(torch)

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
        frame = implementation.Image.fromarray(array, "RGB")
        alpha = implementation.Image.blend(first.getchannel("A"), second.getchannel("A"), t)
        output.append((t, implementation._restore_frame(frame, placement0, first.size, alpha)))
    del clip
    return output


implementation._generate_clip = _generate_clip_offloaded
main = implementation.main


if __name__ == "__main__":
    main()
