#!/usr/bin/env python3
"""Export MoG-VFI into browser-oriented ONNX components.

Python/PyTorch are used only for this offline conversion step. The browser owns
Gaussian posterior sampling, latent motion warping, the v-prediction DDIM loop,
component offload and WebGPU/WASM provider selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MoG browser ONNX components")
    parser.add_argument("--source", type=Path, required=True, help="MCG-NJU/MoG-VFI checkout")
    parser.add_argument("--config", type=Path, required=True, help="MoG ani.yaml or real.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="MoG ani.ckpt or real.ckpt")
    parser.add_argument("--flow-checkpoint", type=Path, required=True, help="EMA-VFI ours_t.ckpt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def require_shape(height: int, width: int, frames: int) -> None:
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("MoG export height/width must be positive multiples of 16")
    if frames < 3:
        raise ValueError("MoG export requires at least three frames")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_flow_checkpoint(source: Path, supplied: Path) -> Path:
    expected = source / "emavfi" / "ckpt" / "ours_t.ckpt"
    expected.parent.mkdir(parents=True, exist_ok=True)
    if supplied.resolve() != expected.resolve():
        if not expected.is_file() or expected.stat().st_size != supplied.stat().st_size:
            shutil.copy2(supplied, expected)
    return expected


def load_checkpoint(torch, model, checkpoint: Path):
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
            renamed = OrderedDict(state_dict)
            for key in list(renamed.keys()):
                if "framestride_embed" in key:
                    renamed[key.replace("framestride_embed", "fps_embedding")] = renamed.pop(key)
            model.load_state_dict(renamed, strict=True)
    else:
        module = state["module"]
        renamed = OrderedDict((key[16:], value) for key, value in module.items())
        model.load_state_dict(renamed, strict=True)
    del state
    gc.collect()
    return model


def export_graph(torch, module, inputs, path: Path, input_names, output_names, dynamic_axes, opset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )
    signature = inspect.signature(torch.onnx.export)
    if "use_external_data_format" in signature.parameters:
        kwargs["use_external_data_format"] = True
    print(f"Exporting {path.name}", flush=True)
    with torch.inference_mode():
        torch.onnx.export(module.eval(), inputs, str(path), **kwargs)


def externalise_onnx(onnx, path: Path) -> list[Path]:
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    data_name = f"{path.stem}.data"
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024 * 1024,
        convert_attribute=False,
    )
    model_check = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model_check)
    files = [path]
    data_path = path.with_name(data_name)
    if data_path.is_file():
        files.append(data_path)
    return files


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    flow_checkpoint = args.flow_checkpoint.resolve()
    output = args.output.resolve()
    require_shape(args.height, args.width, args.frames)
    for path in (source, config_path, checkpoint, flow_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.chdir(source)
    sys.path.insert(0, str(source))
    ensure_flow_checkpoint(source, flow_checkpoint)

    import onnx
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange
    from omegaconf import OmegaConf
    from emavfi.Trainer import Model as VFIModel
    from emavfi.model.warplayer import warp
    from utils.utils import instantiate_from_config

    original_device = VFIModel.device
    VFIModel.device = lambda self: None
    try:
        config = OmegaConf.load(str(config_path))
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
    finally:
        VFIModel.device = original_device

    model = load_checkpoint(torch, model, checkpoint).float().cpu().eval().requires_grad_(False)
    model.perframe_ae = True
    if not hasattr(model, "vfi") or not hasattr(model.vfi, "net"):
        raise RuntimeError("MoG checkpoint did not instantiate EMA-VFI motion guidance")
    vfi_net = model.vfi.net.float().cpu().eval().requires_grad_(False)

    class MotionWrapper(nn.Module):
        def __init__(self, net, frame_count: int):
            super().__init__()
            self.net = net
            self.frame_count = frame_count

        def forward(self, first, second):
            # Mirrors emavfi.vfi_utils.cal_flow. Input endpoints are in [-1, 1].
            first_01 = first.flip(1) * 0.5 + 0.5
            second_01 = second.flip(1) * 0.5 + 0.5
            batch = first.shape[0]
            repeats = self.frame_count - 2
            img0 = first_01.repeat_interleave(repeats=repeats, dim=0)
            img1 = second_01.repeat_interleave(repeats=repeats, dim=0)
            timestep = torch.linspace(0, 1, self.frame_count, device=first.device)[1:-1]
            timestep = timestep.repeat(batch).reshape(batch * repeats, 1, 1, 1)
            flow = None
            mask = None
            appearance_features, motion_features = self.net.feature_bone(img0, img1)
            half_batch = appearance_features[0].shape[0] // 2
            for stage in range(self.net.flow_num_stage):
                t = torch.ones_like(motion_features[-1-stage][:half_batch]) * timestep
                if flow is None:
                    flow, mask = self.net.block[stage](
                        torch.cat([
                            t * motion_features[-1-stage][:half_batch],
                            (1-t) * motion_features[-1-stage][half_batch:],
                            appearance_features[-1-stage][:half_batch],
                            appearance_features[-1-stage][half_batch:],
                        ], 1),
                        torch.cat((img0, img1), 1),
                        None,
                    )
                else:
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    flow_delta, mask_delta = self.net.block[stage](
                        torch.cat([
                            t * motion_features[-1-stage][:half_batch],
                            (1-t) * motion_features[-1-stage][half_batch:],
                            appearance_features[-1-stage][:half_batch],
                            appearance_features[-1-stage][half_batch:],
                        ], 1),
                        torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                        flow,
                    )
                    flow = flow + flow_delta
                    mask = mask + mask_delta
            return flow, torch.sigmoid(mask)

    class VaeEncoderWrapper(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.first_stage_model = parent.first_stage_model

        def forward(self, video):
            batch, channels, frames, height, width = video.shape
            flat = rearrange(video, "b c t h w -> (b t) c h w")
            posterior, hidden_states = self.first_stage_model.encode(flat, return_hidden_states=True)
            parameters = rearrange(
                posterior.parameters,
                "(b t) c h w -> b c t h w",
                b=batch,
                t=frames,
            )
            context = []
            for hidden in hidden_states:
                hidden_5d = rearrange(hidden, "(b t) c h w -> b c t h w", b=batch, t=frames)
                context.append(torch.cat([hidden_5d[:, :, 0:1], hidden_5d[:, :, -1:]], dim=2))
            return (parameters, *context)

    empty_text = model.get_learned_conditioning([""]).detach().float().cpu()

    class ImageConditionWrapper(nn.Module):
        def __init__(self, parent, empty_condition):
            super().__init__()
            self.embedder = parent.embedder
            self.projector = parent.image_proj_model
            self.register_buffer("empty_condition", empty_condition)

        def forward(self, first):
            image_embedding = self.projector(self.embedder(first))
            text = self.empty_condition.expand(first.shape[0], -1, -1)
            return torch.cat([text, image_embedding], dim=1)

    class DenoiserWrapper(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.diffusion = parent.model

        def forward(self, sample, timestep, cross_attention, concat_latent, flow, mask, fps):
            return self.diffusion(
                sample,
                timestep,
                c_concat=[concat_latent],
                c_crossattn=[cross_attention],
                motion_guidance=[flow, mask],
                fs=fps,
            )

    class DecoderWrapper(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def forward(self, latent, *reference_context):
            return self.parent.decode_first_stage(latent, ref_context=list(reference_context))

    first = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    second = torch.zeros_like(first)
    video = torch.cat([
        first.unsqueeze(2).repeat(1, 1, args.frames // 2, 1, 1),
        second.unsqueeze(2).repeat(1, 1, args.frames - args.frames // 2, 1, 1),
    ], dim=2)

    output.mkdir(parents=True, exist_ok=True)
    graph_paths: list[Path] = []

    motion_path = output / "motion.onnx"
    export_graph(
        torch,
        MotionWrapper(vfi_net, args.frames),
        (first, second),
        motion_path,
        ["first", "second"],
        ["flow", "mask"],
        {
            "first": {0: "batch", 2: "height", 3: "width"},
            "second": {0: "batch", 2: "height", 3: "width"},
            "flow": {0: "interior_frames", 2: "height", 3: "width"},
            "mask": {0: "interior_frames", 2: "height", 3: "width"},
        },
        args.opset,
    )
    graph_paths.append(motion_path)

    encoder_wrapper = VaeEncoderWrapper(model)
    with torch.inference_mode():
        encoder_example = encoder_wrapper(video)
    hidden_count = len(encoder_example) - 1
    encoder_outputs = ["posterior_parameters"] + [f"reference_hidden_{index}" for index in range(hidden_count)]
    encoder_axes = {
        "video": {0: "batch", 3: "height", 4: "width"},
        "posterior_parameters": {0: "batch", 3: "latent_height", 4: "latent_width"},
    }
    for name in encoder_outputs[1:]:
        encoder_axes[name] = {0: "batch", 3: f"{name}_height", 4: f"{name}_width"}
    encoder_path = output / "vae_encoder.onnx"
    export_graph(torch, encoder_wrapper, (video,), encoder_path, ["video"], encoder_outputs, encoder_axes, args.opset)
    graph_paths.append(encoder_path)

    conditioning_path = output / "image_condition.onnx"
    export_graph(
        torch,
        ImageConditionWrapper(model, empty_text),
        (first,),
        conditioning_path,
        ["first"],
        ["cross_attention"],
        {
            "first": {0: "batch", 2: "height", 3: "width"},
            "cross_attention": {0: "batch"},
        },
        args.opset,
    )
    graph_paths.append(conditioning_path)

    with torch.inference_mode():
        posterior_parameters = encoder_example[0]
        latent_channels = posterior_parameters.shape[1] // 2
        latent_height = posterior_parameters.shape[3]
        latent_width = posterior_parameters.shape[4]
        cross_attention = ImageConditionWrapper(model, empty_text)(first)
        flow, mask = MotionWrapper(vfi_net, args.frames)(first, second)
    sample = torch.zeros(1, latent_channels, args.frames, latent_height, latent_width, dtype=torch.float32)
    concat_latent = torch.zeros_like(sample)
    timestep = torch.zeros(1, dtype=torch.long)
    fps = torch.full((1,), 24, dtype=torch.long)
    denoiser_path = output / "denoiser.onnx"
    export_graph(
        torch,
        DenoiserWrapper(model),
        (sample, timestep, cross_attention, concat_latent, flow, mask, fps),
        denoiser_path,
        ["sample", "timestep", "cross_attention", "concat_latent", "flow", "mask", "fps"],
        ["velocity"],
        {
            "sample": {0: "batch", 3: "latent_height", 4: "latent_width"},
            "timestep": {0: "batch"},
            "cross_attention": {0: "batch"},
            "concat_latent": {0: "batch", 3: "latent_height", 4: "latent_width"},
            "flow": {0: "interior_frames", 2: "height", 3: "width"},
            "mask": {0: "interior_frames", 2: "height", 3: "width"},
            "fps": {0: "batch"},
            "velocity": {0: "batch", 3: "latent_height", 4: "latent_width"},
        },
        args.opset,
    )
    graph_paths.append(denoiser_path)

    latent_example = torch.zeros(1, latent_channels, args.frames, latent_height, latent_width, dtype=torch.float32)
    reference_context = tuple(encoder_example[1:])
    decoder_path = output / "vae_decoder.onnx"
    decoder_input_names = ["latent"] + [f"reference_hidden_{index}" for index in range(hidden_count)]
    decoder_axes = {
        "latent": {0: "batch", 3: "latent_height", 4: "latent_width"},
        "decoded": {0: "batch", 3: "height", 4: "width"},
    }
    for name in decoder_input_names[1:]:
        decoder_axes[name] = {0: "batch", 3: f"{name}_height", 4: f"{name}_width"}
    export_graph(
        torch,
        DecoderWrapper(model),
        (latent_example, *reference_context),
        decoder_path,
        decoder_input_names,
        ["decoded"],
        decoder_axes,
        args.opset,
    )
    graph_paths.append(decoder_path)

    asset_files: list[Path] = []
    for graph in graph_paths:
        asset_files.extend(externalise_onnx(onnx, graph))

    assets = []
    for path in sorted(set(asset_files)):
        assets.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    manifest = {
        "format": 1,
        "family": "mog",
        "variant": config_path.stem,
        "source": "MCG-NJU/MoG-VFI",
        "components": {
            "motion": "motion.onnx",
            "vaeEncoder": "vae_encoder.onnx",
            "imageCondition": "image_condition.onnx",
            "denoiser": "denoiser.onnx",
            "vaeDecoder": "vae_decoder.onnx",
        },
        "frames": args.frames,
        "posteriorScaleFactor": float(model.scale_factor),
        "referenceHiddenCount": hidden_count,
        "parameterization": str(model.parameterization),
        "fps": 24,
        "ddim": {
            "trainingTimesteps": 1000,
            "inferenceSteps": 50,
            "linearStart": 0.00085,
            "linearEnd": 0.012,
            "eta": 1.0,
            "zeroTerminalSnr": True,
            "spacing": "uniform_trailing",
        },
        "opset": args.opset,
        "assets": assets,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
