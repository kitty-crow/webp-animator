#!/usr/bin/env python3
"""Export ToonCrafter into browser-oriented ONNX components.

The browser retains VAE posterior sampling, endpoint latent construction,
v-prediction DDIM scheduling/dynamic rescaling and component offload. Python and
PyTorch are used only by this offline conversion utility.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ToonCrafter browser ONNX components")
    parser.add_argument("--source", type=Path, required=True, help="Doubiiu/ToonCrafter checkout")
    parser.add_argument("--config", type=Path, required=True, help="inference_512_v1.0.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="ToonCrafter model.ckpt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def validate_shape(height: int, width: int) -> None:
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("ToonCrafter export dimensions must be positive multiples of 16")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(torch, model, checkpoint: Path):
    kwargs = {"map_location": "cpu"}
    try:
        state = torch.load(str(checkpoint), mmap=True, **kwargs)
    except TypeError:
        state = torch.load(str(checkpoint), **kwargs)
    state_dict = state.get("state_dict", state)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        renamed = OrderedDict(state_dict)
        for key in list(renamed.keys()):
            if "framestride_embed" in key:
                renamed[key.replace("framestride_embed", "fps_embedding")] = renamed.pop(key)
        model.load_state_dict(renamed, strict=True)
    del state, state_dict
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
    if "use_external_data_format" in inspect.signature(torch.onnx.export).parameters:
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
    check = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(check)
    files = [path]
    external = path.with_name(data_name)
    if external.is_file():
        files.append(external)
    return files


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    validate_shape(args.height, args.width)
    for path in (source, config_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "lvdm"))

    import onnx
    import torch
    import torch.nn as nn
    from einops import rearrange
    from omegaconf import OmegaConf
    from utils.utils import instantiate_from_config

    config = OmegaConf.load(str(config_path))
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    model = instantiate_from_config(model_config)
    model = load_checkpoint(torch, model, checkpoint).float().cpu().eval().requires_grad_(False)
    model.perframe_ae = True
    frame_count = int(model.temporal_length)

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

        def forward(self, sample, timestep, cross_attention, concat_latent, fps):
            return self.diffusion(
                sample,
                timestep,
                c_concat=[concat_latent],
                c_crossattn=[cross_attention],
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
    split = frame_count // 2
    video = torch.cat([
        first.unsqueeze(2).repeat(1, 1, split, 1, 1),
        second.unsqueeze(2).repeat(1, 1, frame_count - split, 1, 1),
    ], dim=2)
    output.mkdir(parents=True, exist_ok=True)
    graphs: list[Path] = []

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
    graphs.append(encoder_path)

    condition_wrapper = ImageConditionWrapper(model, empty_text)
    with torch.inference_mode():
        cross_attention = condition_wrapper(first)
    condition_path = output / "image_condition.onnx"
    export_graph(
        torch,
        condition_wrapper,
        (first,),
        condition_path,
        ["first"],
        ["cross_attention"],
        {"first": {0: "batch", 2: "height", 3: "width"}, "cross_attention": {0: "batch"}},
        args.opset,
    )
    graphs.append(condition_path)

    posterior_parameters = encoder_example[0]
    latent_channels = int(posterior_parameters.shape[1] // 2)
    latent_height = int(posterior_parameters.shape[3])
    latent_width = int(posterior_parameters.shape[4])
    sample = torch.zeros(1, latent_channels, frame_count, latent_height, latent_width, dtype=torch.float32)
    concat_latent = torch.zeros_like(sample)
    timestep = torch.zeros(1, dtype=torch.long)
    fps = torch.full((1,), 24, dtype=torch.long)
    denoiser_path = output / "denoiser.onnx"
    export_graph(
        torch,
        DenoiserWrapper(model),
        (sample, timestep, cross_attention, concat_latent, fps),
        denoiser_path,
        ["sample", "timestep", "cross_attention", "concat_latent", "fps"],
        ["velocity"],
        {
            "sample": {0: "batch", 3: "latent_height", 4: "latent_width"},
            "timestep": {0: "batch"},
            "cross_attention": {0: "batch"},
            "concat_latent": {0: "batch", 3: "latent_height", 4: "latent_width"},
            "fps": {0: "batch"},
            "velocity": {0: "batch", 3: "latent_height", 4: "latent_width"},
        },
        args.opset,
    )
    graphs.append(denoiser_path)

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
        (sample, *tuple(encoder_example[1:])),
        decoder_path,
        decoder_input_names,
        ["decoded"],
        decoder_axes,
        args.opset,
    )
    graphs.append(decoder_path)

    asset_files: list[Path] = []
    for graph in graphs:
        asset_files.extend(externalise_onnx(onnx, graph))
    assets = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(set(asset_files))
    ]
    manifest = {
        "format": 1,
        "family": "tooncrafter",
        "source": "Doubiiu/ToonCrafter",
        "components": {
            "vaeEncoder": "vae_encoder.onnx",
            "imageCondition": "image_condition.onnx",
            "denoiser": "denoiser.onnx",
            "vaeDecoder": "vae_decoder.onnx",
        },
        "frames": frame_count,
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
            "baseScale": float(model.base_scale),
            "turningStep": 400,
        },
        "opset": args.opset,
        "assets": assets,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
