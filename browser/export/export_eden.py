#!/usr/bin/env python3
"""Export EDEN's native two-step midpoint path as one browser ONNX graph."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export EDEN browser ONNX graph")
    parser.add_argument("--source", type=Path, required=True, help="bbldCVer/EDEN checkout")
    parser.add_argument("--config", type=Path, required=True, help="eval_eden.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="eden.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def externalise(onnx, path: Path) -> list[Path]:
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
    onnx.checker.check_model(onnx.load(str(path), load_external_data=False))
    files = [path]
    data = path.with_name(data_name)
    if data.is_file():
        files.append(data)
    return files


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    for path in (source, config_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)

    height = max(32, round(args.height / 32) * 32)
    width = max(32, round(args.width / 32) * 32)
    os.chdir(source)
    sys.path.insert(0, str(source))

    import onnx
    import torch
    import torch.nn as nn
    import yaml
    from src.models import load_model

    config = yaml.unsafe_load(config_path.read_text(encoding="utf-8"))
    model_args = dict(config["model_args"])
    model_args["use_xformers"] = False
    model = load_model(config["model_name"], **model_args)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint["eden"])
    model = model.cpu().eval().requires_grad_(False)
    vae_scaler = float(config["vae_scaler"])
    vae_shift = float(config["vae_shift"])
    latent_dim = int(model_args["latent_dim"])

    class EdenWrapper(nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, first, second, noise, difference):
            cond_frames = torch.cat((first, second), dim=0)
            batch = first.shape[0]
            t0 = torch.zeros((batch,), dtype=first.dtype, device=first.device)
            velocity0 = self.network.denoise(noise, t0, cond_frames, difference)
            sample = noise + 0.75 * velocity0
            t1 = torch.full((batch,), 0.75, dtype=first.dtype, device=first.device)
            velocity1 = self.network.denoise(sample, t1, cond_frames, difference)
            sample = sample + 0.25 * velocity1
            latents = sample / vae_scaler + vae_shift
            return self.network.decode(latents).clamp(0.0, 1.0)

    first = torch.zeros(1, 3, height, width, dtype=torch.float32)
    second = torch.zeros_like(first)
    noise = torch.zeros(1, (height // 32) * (width // 32), latent_dim, dtype=torch.float32)
    difference = torch.zeros(1, 1, dtype=torch.float32)
    graph = output / "eden_midpoint.onnx"
    output.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["first", "second", "noise", "difference"],
        output_names=["midpoint"],
        opset_version=args.opset,
        do_constant_folding=True,
        export_params=True,
    )
    if "use_external_data_format" in inspect.signature(torch.onnx.export).parameters:
        kwargs["use_external_data_format"] = True
    with torch.inference_mode():
        torch.onnx.export(EdenWrapper(model).eval(), (first, second, noise, difference), str(graph), **kwargs)
    files = externalise(onnx, graph)

    manifest = {
        "format": 1,
        "family": "eden",
        "source": "bbldCVer/EDEN",
        "licence": "Apache-2.0",
        "components": {"generator": graph.name},
        "internalWidth": width,
        "internalHeight": height,
        "latentDim": latent_dim,
        "cosSimMean": float(config["cos_sim_mean"]),
        "cosSimStd": float(config["cos_sim_std"]),
        "seed": 0,
        "solver": {"method": "euler", "times": [0.0, 0.75, 1.0]},
        "assets": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(set(files))
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
