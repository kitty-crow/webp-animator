#!/usr/bin/env python3
"""Export the SPEED midpoint generator to a browser ONNX graph."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SPEED browser ONNX graph")
    parser.add_argument("--source", type=Path, required=True, help="bbldCVer/SPEED checkout")
    parser.add_argument("--config", type=Path, required=True, help="eval_config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="speed.pt")
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


def export_graph(torch, module, inputs, path: Path, opset: int) -> None:
    kwargs = dict(
        input_names=["first", "second", "noise"],
        output_names=["midpoint"],
        dynamic_axes={
            "first": {0: "batch", 2: "height", 3: "width"},
            "second": {0: "batch", 2: "height", 3: "width"},
            "noise": {0: "batch", 2: "height", 3: "width"},
            "midpoint": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )
    if "use_external_data_format" in inspect.signature(torch.onnx.export).parameters:
        kwargs["use_external_data_format"] = True
    with torch.inference_mode():
        torch.onnx.export(module.eval(), inputs, str(path), **kwargs)


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
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    for path in (source, config, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.height <= 0 or args.width <= 0:
        raise ValueError("SPEED export dimensions must be positive")

    os.chdir(source)
    sys.path.insert(0, str(source))
    import onnx
    import torch
    import torch.nn as nn
    import inference as speed_inference

    model = speed_inference.build_model(
        str(config),
        str(checkpoint),
        torch.device("cpu"),
        strict_load=False,
    ).cpu().eval().requires_grad_(False)

    class SpeedWrapper(nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, first, second, noise):
            cond_frames = torch.cat((first * 2.0 - 1.0, second * 2.0 - 1.0), dim=0)
            timestep = torch.full((first.shape[0],), 1000.0, dtype=first.dtype, device=first.device)
            prediction = self.network(
                noisy_frames=noise,
                cond_frames=cond_frames,
                timestep=timestep,
            )
            return (prediction / 2.0 + 0.5).clamp(0.0, 1.0)

    height = max(32, round(args.height / 16) * 16)
    width = max(32, round(args.width / 16) * 16)
    first = torch.zeros(1, 3, height, width, dtype=torch.float32)
    second = torch.zeros_like(first)
    noise = torch.zeros_like(first)
    output.mkdir(parents=True, exist_ok=True)
    graph = output / "speed_midpoint.onnx"
    files = externalise(onnx, graph) if False else []
    export_graph(torch, SpeedWrapper(model), (first, second, noise), graph, args.opset)
    files = externalise(onnx, graph)

    manifest = {
        "format": 1,
        "family": "speed",
        "source": "bbldCVer/SPEED",
        "licence": "Upstream repository does not publish a root licence file; verify checkpoint redistribution terms before hosting exported weights.",
        "components": {"generator": graph.name},
        "scales": [1.0, 0.75, 0.5, 0.375],
        "seed": 0,
        "assets": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(set(files))
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
