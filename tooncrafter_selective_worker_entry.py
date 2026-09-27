#!/usr/bin/env python3
from __future__ import annotations

import tooncrafter_selective_worker as implementation
from model_offload import apply_model_offload


# The official checkpoint is roughly 10.5 GB. Loading it into a freshly-created
# FP32 model before converting that model to FP16 creates an unnecessary host-RAM
# peak. Allocate destination parameters in FP16 before load_state_dict copies the
# checkpoint into them.
_original_load_checkpoint = implementation._load_checkpoint_mmap


def _load_checkpoint_low_ram(torch, model, checkpoint):
    model = model.half().eval()
    return _original_load_checkpoint(torch, model, checkpoint)


implementation._load_checkpoint_mmap = _load_checkpoint_low_ram


# Keep source/checkpoint metadata on the model so _move_core can choose between
# full GPU residency, RAM streaming and memory-mapped disk streaming when the
# denoiser is first needed.
_original_load_model = implementation.load_model


def _load_model_offload_ready(source, config_path, checkpoint):
    model_tuple = _original_load_model(source, config_path, checkpoint)
    torch, device, model, total_vram = model_tuple
    setattr(model, "_webp_source", source)
    setattr(model, "_webp_checkpoint", checkpoint)
    setattr(model, "_webp_execution_device", device)
    return torch, device, model, total_vram


implementation.load_model = _load_model_offload_ready


_original_move_core = implementation._move_core


def _move_core_offloaded(model, device):
    import torch

    target = torch.device(device)
    current_mode = getattr(model, "_webp_offload_mode", None)

    # RAM/disk hooks already load each layer onto CUDA only for the forward that
    # needs it, then evict it again. A later request to "move core to CPU" before
    # VAE decode therefore requires no full-model transfer.
    if current_mode in {"cpu", "disk"}:
        return model

    if target.type != "cuda":
        return _original_move_core(model, target)

    source = getattr(model, "_webp_source", None)
    checkpoint = getattr(model, "_webp_checkpoint", None)
    if source is None or checkpoint is None:
        return _original_move_core(model, target)

    # Auxiliaries are already staged separately by the ToonCrafter worker. Keep
    # them out of the offload hook graph so VAE/text/image components can still be
    # moved independently at their existing call sites.
    auxiliaries = implementation._auxiliary_modules(model)
    for name, module in auxiliaries.items():
        if module is not None:
            setattr(model, name, None)
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
main = implementation.main


if __name__ == "__main__":
    main()
