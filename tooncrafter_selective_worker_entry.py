#!/usr/bin/env python3
from __future__ import annotations

import gc

import tooncrafter_selective_worker as implementation
from model_offload import apply_model_offload


SPATIAL_ALIGNMENT = 64


def _normalise_target(target):
    """Keep ToonCrafter's latent U-Net skip geometry exact.

    The VAE downsamples by 8 and the four-level U-Net downsamples the latent three
    more times, so both input dimensions must be multiples of 64.  Older low-VRAM
    rungs such as 224x128 or 256x160 create odd latent sizes and eventually make
    skip tensors differ by one pixel.
    """
    width, height = (max(SPATIAL_ALIGNMENT, int(value)) for value in target)
    aligned = (
        max(SPATIAL_ALIGNMENT * 2, (width // SPATIAL_ALIGNMENT) * SPATIAL_ALIGNMENT),
        max(SPATIAL_ALIGNMENT * 2, (height // SPATIAL_ALIGNMENT) * SPATIAL_ALIGNMENT),
    )
    if aligned != (width, height):
        print(
            f"ToonCrafter target {width}x{height} is not U-Net aligned; "
            f"using {aligned[0]}x{aligned[1]} instead",
            flush=True,
        )
    return aligned


def _target_ladder(total_vram):
    gb = total_vram / 1024**3
    if gb <= 4.5:
        return [(256, 128), (192, 128), (128, 128)]
    if gb <= 8.0:
        return [(320, 192), (256, 128), (192, 128), (128, 128)]
    if gb <= 12.0:
        return [(448, 256), (384, 256), (320, 192), (256, 128)]
    return [(512, 320), (448, 256), (384, 256), (320, 192), (256, 128), (192, 128)]


implementation._target_ladder = _target_ladder


# The official checkpoint is roughly 10.5 GB. Loading it into a freshly-created
# FP32 model before converting that model to FP16 creates an unnecessary host-RAM
# peak. Allocate destination parameters in FP16 before load_state_dict copies the
# checkpoint into them.
_original_load_checkpoint = implementation._load_checkpoint_mmap


def _load_checkpoint_low_ram(torch, model, checkpoint):
    model.half()
    # Do not chain .eval(): a few research dependencies replace train() with helpers
    # whose return value is not the module, which makes Module.eval() return bool.
    torch.nn.Module.train(model, False)
    return _original_load_checkpoint(torch, model, checkpoint)


implementation._load_checkpoint_mmap = _load_checkpoint_low_ram


# Keep source/checkpoint metadata on the model so _move_core can choose between
# full GPU residency, RAM streaming and memory-mapped disk streaming when the
# denoiser is first needed.
_original_load_model = implementation.load_model


def _repair_bad_eval_overrides(torch, model):
    """Remove only instance train() overrides that make eval() return non-modules."""
    modules = [model]
    modules.extend(
        module for module in implementation._auxiliary_modules(model).values()
        if isinstance(module, torch.nn.Module)
    )
    repaired = 0
    for module in modules:
        result = module.eval()
        if isinstance(result, torch.nn.Module):
            continue
        if "train" in module.__dict__:
            del module.__dict__["train"]
        torch.nn.Module.train(module, False)
        repaired += 1
    if repaired:
        print(f"ToonCrafter repaired {repaired} non-standard train/eval override(s)", flush=True)


def _load_model_offload_ready(source, config_path, checkpoint):
    import torch

    original_module_eval = torch.nn.Module.eval

    def stable_module_eval(module):
        # Several video-research dependencies monkey-patch train().  Calling the base
        # implementation directly guarantees eval() always returns the module while
        # preserving the intended inference state.
        torch.nn.Module.train(module, False)
        return module

    torch.nn.Module.eval = stable_module_eval
    try:
        model_tuple = _original_load_model(source, config_path, checkpoint)
    finally:
        torch.nn.Module.eval = original_module_eval

    torch_module, device, model, total_vram = model_tuple
    _repair_bad_eval_overrides(torch_module, model)
    setattr(model, "_webp_source", source)
    setattr(model, "_webp_checkpoint", checkpoint)
    setattr(model, "_webp_execution_device", device)
    return torch_module, device, model, total_vram


implementation.load_model = _load_model_offload_ready


_original_move_core = implementation._move_core


def _move_core_offloaded(model, device):
    import torch

    target = torch.device(device)
    current_mode = getattr(model, "_webp_offload_mode", None)

    # RAM/disk hooks already load each layer onto CUDA only for the forward that
    # needs it, then evict it again. A later request to move the core to CPU before
    # VAE decode therefore requires no full-model transfer.
    if current_mode in {"cpu", "disk"}:
        return model

    if target.type != "cuda":
        return _original_move_core(model, target)

    source = getattr(model, "_webp_source", None)
    checkpoint = getattr(model, "_webp_checkpoint", None)
    if source is None or checkpoint is None:
        return _original_move_core(model, target)

    # Auxiliaries are staged separately by the ToonCrafter worker. Keep them out of
    # the offload graph so VAE/text/image components retain their existing explicit
    # lifecycle. The training-time EMA shadow is not used by this inference path and
    # would otherwise roughly duplicate the denoiser's host/disk footprint.
    auxiliaries = implementation._auxiliary_modules(model)
    for name, module in auxiliaries.items():
        if module is not None:
            setattr(model, name, None)

    if getattr(model, "model_ema", None) is not None:
        model.model_ema = None
        if hasattr(model, "use_ema"):
            model.use_ema = False
        gc.collect()

    try:
        model, _mode, _stats = apply_model_offload(
            model,
            target,
            source=source,
            checkpoint=checkpoint,
            env_var="TOONCRAFTER_OFFLOAD",
            namespace="tooncrafter",
        )
    finally:
        for name, module in auxiliaries.items():
            if module is not None:
                setattr(model, name, module)
    return model


implementation._move_core = _move_core_offloaded


_original_generate_clip = implementation._generate_clip


def _generate_clip_aligned(torch, device, model, first, second, target, *, ddim_steps):
    return _original_generate_clip(
        torch,
        device,
        model,
        first,
        second,
        _normalise_target(target),
        ddim_steps=ddim_steps,
    )


implementation._generate_clip = _generate_clip_aligned
main = implementation.main


if __name__ == "__main__":
    main()
