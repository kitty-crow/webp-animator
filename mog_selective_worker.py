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

    original_torch_load = torch.load

    def cpu_default_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("map_location", "cpu")
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = cpu_default_load
    try:
        return _original_load_model(*args, **kwargs)
    finally:
        torch.load = original_torch_load


implementation.load_model = _load_model_cpu_safe
main = implementation.main


if __name__ == "__main__":
    main()
