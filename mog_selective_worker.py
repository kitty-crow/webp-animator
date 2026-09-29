#!/usr/bin/env python3
from __future__ import annotations

import gc
import shutil

import mog_selective_worker_v2 as implementation
from model_offload import apply_model_offload


SPATIAL_ALIGNMENT = 64


def _normalise_target(target):
    """Return a spatial size compatible with MoG's VAE + four-level U-Net.

    The VAE contributes an 8x reduction and the U-Net downsamples latent space three
    more times. Both source dimensions therefore need to be multiples of 64 or skip
    tensors can differ by one pixel during the up path (for example 8 versus 7).
    """
    width, height = (max(SPATIAL_ALIGNMENT, int(value)) for value in target)
    aligned = (
        max(SPATIAL_ALIGNMENT * 2, (width // SPATIAL_ALIGNMENT) * SPATIAL_ALIGNMENT),
        max(SPATIAL_ALIGNMENT * 2, (height // SPATIAL_ALIGNMENT) * SPATIAL_ALIGNMENT),
    )
    if aligned != (width, height):
        print(
            f"MoG target {width}x{height} is not U-Net aligned; "
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


# Capture the exact top-level MoG model at checkpoint-load time. This is critical:
# OpenCLIP and other large children call .to(...) while the parent model is still
# being constructed. The old size-based global Module.to hook could mistake one of
# those children for the diffusion model, offload it to meta storage too early, and
# then make the subsequent MoG checkpoint load a no-op for those parameters.
_top_level_model = {"value": None}


# The released MoG checkpoints are roughly ten gigabytes in FP32. Convert the
# freshly instantiated parameter storage to FP16 before load_state_dict copies the
# mmap-backed checkpoint into it. This avoids holding a full-size FP32 model plus a
# full-size checkpoint resident in host RAM on 16 GB machines.
_original_load_checkpoint = implementation._load_checkpoint_mmap


def _load_checkpoint_low_ram(torch, model, checkpoint):
    _top_level_model["value"] = model
    model = model.half().eval()
    loaded = _original_load_checkpoint(torch, model, checkpoint)

    # A freshly constructed model must not contain meta placeholders here. Meta is
    # only legal after WebP Animator deliberately installs Accelerate offload hooks.
    # Catch accidental early-offload regressions before model placement.
    meta = [
        name
        for name, tensor in list(loaded.named_parameters()) + list(loaded.named_buffers())
        if getattr(getattr(tensor, "device", None), "type", None) == "meta"
    ]
    if meta:
        preview = ", ".join(meta[:6])
        raise RuntimeError(
            "MoG checkpoint load left parameters on the meta device before offload "
            f"was installed ({len(meta)} tensors; first: {preview}). This indicates "
            "an invalid model-construction/offload ordering."
        )
    return loaded


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


# Upstream EMA-VFI deliberately replaces `net.train` with `lambda x: x` after
# putting the flow network in eval mode. Unfortunately torch.nn.Module.eval()
# is implemented as `return self.train(False)`, so any later chained `.eval()`
# returns the boolean False instead of the module. The low-VRAM staging code does
# exactly that while moving the flow network CPU <-> CUDA. During construction we
# therefore use an eval implementation with normal Module semantics, then remove the
# upstream instance-level train override once the model is fully loaded.
def _repair_vfi_train_override(torch, model) -> None:
    vfi = getattr(model, "vfi", None)
    net = getattr(vfi, "net", None)
    if not isinstance(net, torch.nn.Module):
        raise RuntimeError(
            "MoG EMA-VFI motion network was not a torch.nn.Module after model load; "
            f"got {type(net).__name__}."
        )
    if "train" in net.__dict__:
        del net.__dict__["train"]
    net.eval()
    if not isinstance(getattr(vfi, "net", None), torch.nn.Module):
        raise RuntimeError("MoG EMA-VFI motion network became invalid while restoring eval semantics.")


# Upstream performs one top-level `model.to(cuda)` after the checkpoint is loaded.
# Intercept exactly that object, not arbitrary large children. Low-VRAM machines
# then use Accelerate CPU/disk offload; ordinary construction-time .to() calls on
# OpenCLIP, VAE, flow, etc. retain normal PyTorch semantics.
_original_load_model = implementation.load_model


def _load_model_offloaded(source, config_path, checkpoint, flow_checkpoint):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MoG requires a working CUDA runtime.")

    original_torch_load = torch.load
    original_module_to = torch.nn.Module.to
    original_module_eval = torch.nn.Module.eval
    routed = {"done": False}

    def cpu_default_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("map_location", "cpu")
        return original_torch_load(*load_args, **load_kwargs)

    def stable_module_eval(module):
        # Call the class implementation directly so an instance-level `train`
        # replacement (as used by MoG's EMA-VFI helper) cannot change eval()'s
        # return type from Module to bool.
        torch.nn.Module.train(module, False)
        return module

    def routed_module_to(module, *to_args, **to_kwargs):
        target = _top_level_model.get("value")
        if not routed["done"] and target is not None and module is target:
            # MoG inference never enters ema_scope(), so the training-time EMA shadow
            # is dead weight here and is roughly another model-sized set of buffers.
            if getattr(module, "model_ema", None) is not None:
                module.model_ema = None
                if hasattr(module, "use_ema"):
                    module.use_ema = False
                gc.collect()

            routed["done"] = True
            # apply_model_offload's fully-resident path itself calls model.to(cuda).
            # Temporarily restore PyTorch's real method so that call cannot recurse
            # back into this interception hook.
            torch.nn.Module.to = original_module_to
            try:
                placed, _mode, _stats = apply_model_offload(
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
    torch.nn.Module.eval = stable_module_eval
    _top_level_model["value"] = None
    try:
        model_tuple = _original_load_model(source, config_path, checkpoint, flow_checkpoint)
        if not routed["done"]:
            raise RuntimeError(
                "MoG top-level model placement hook did not run; refusing an "
                "unqualified load path."
            )
        _repair_vfi_train_override(torch, model_tuple[2])
        return model_tuple
    finally:
        _top_level_model["value"] = None
        torch.load = original_torch_load
        torch.nn.Module.to = original_module_to
        torch.nn.Module.eval = original_module_eval


implementation.load_model = _load_model_offloaded


# Accelerate CPU/disk offload leaves actual parameters on CPU/meta between calls.
# The original helper inferred the input device from next(model.parameters()), which
# is therefore no longer reliable. Use the explicit execution device instead.
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
    target = _normalise_target(target)
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
