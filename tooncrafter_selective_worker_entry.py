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
main = implementation.main


if __name__ == "__main__":
    main()
