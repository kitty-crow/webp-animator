#!/usr/bin/env python3
from __future__ import annotations

import shutil

import mog_selective_worker_v2 as implementation


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
_original_load_model = implementation.load_model


def _load_model_cpu_safe(*args, **kwargs):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MoG requires a working CUDA runtime.")

    properties = torch.cuda.get_device_properties(0)
    total_vram = int(properties.total_memory)
    minimum_vram = 5 * 1024**3
    if total_vram < minimum_vram:
        raise RuntimeError(
            "MoG cannot run on this GPU: the FP16 diffusion model itself does not fit "
            f"in {total_vram / 1024**3:.1f} GB of VRAM. This failure happens during "
            "model placement, before any frame-resolution-dependent allocations, so "
            "the reduced-resolution fallback ladder cannot make a 4 GB card work. "
            "Use MoG on a higher-VRAM GPU or choose RIFE, AMT, ResShift, EDEN or SPEED "
            "on this machine."
        )

    original_torch_load = torch.load

    def cpu_default_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("map_location", "cpu")
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = cpu_default_load
    try:
        try:
            return _original_load_model(*args, **kwargs)
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in str(exc).lower():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise RuntimeError(
                    "MoG exhausted GPU memory while placing its diffusion model. "
                    "This is a model-residency limit rather than a frame-resolution "
                    "limit, so retrying at a smaller interpolation resolution cannot "
                    "fix it on this GPU. Use a higher-VRAM GPU for MoG."
                ) from exc
            raise
    finally:
        torch.load = original_torch_load


implementation.load_model = _load_model_cpu_safe
main = implementation.main


if __name__ == "__main__":
    main()
