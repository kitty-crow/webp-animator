#!/usr/bin/env python3
from __future__ import annotations

import tooncrafter_selective_worker as implementation


# The official checkpoint is roughly 10.5 GB. Loading it into a freshly-created
# FP32 model before converting that model to FP16 creates an unnecessary host-RAM
# peak, which is especially hostile to the 16 GB laptop used for compatibility
# testing. Allocate the destination parameters in FP16 first; load_state_dict casts
# checkpoint tensors into that storage as it copies them.
_original_load_checkpoint = implementation._load_checkpoint_mmap


def _load_checkpoint_low_ram(torch, model, checkpoint):
    model = model.half().eval()
    return _original_load_checkpoint(torch, model, checkpoint)


implementation._load_checkpoint_mmap = _load_checkpoint_low_ram


_original_load_model = implementation.load_model


def _load_model_with_vram_guard(*args, **kwargs):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ToonCrafter requires a working CUDA runtime.")

    properties = torch.cuda.get_device_properties(0)
    total_vram = int(properties.total_memory)
    minimum_vram = 5 * 1024**3
    if total_vram < minimum_vram:
        raise RuntimeError(
            "ToonCrafter cannot run on this GPU: its diffusion core has a model-placement "
            f"floor above the detected {total_vram / 1024**3:.1f} GB of VRAM. Reduced "
            "render resolution only lowers activation memory after the model is resident; "
            "it cannot make the core itself fit on a 4 GB card. Use ToonCrafter on a "
            "higher-VRAM GPU."
        )

    try:
        return _original_load_model(*args, **kwargs)
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in str(exc).lower():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            raise RuntimeError(
                "ToonCrafter exhausted GPU memory while preparing its model. This is a "
                "model-residency limit, so lowering interpolation resolution cannot fix "
                "this placement failure on the current GPU."
            ) from exc
        raise


implementation.load_model = _load_model_with_vram_guard
main = implementation.main


if __name__ == "__main__":
    main()
