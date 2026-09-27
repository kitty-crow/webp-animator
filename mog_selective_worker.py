#!/usr/bin/env python3
from __future__ import annotations

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
