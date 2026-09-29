#!/usr/bin/env python3
"""Export the native Multi-Input ResShift checkpoint into browser components.

This script is deliberately an offline build tool. Python and PyTorch are never
part of the GitHub Pages runtime. The browser owns scheduling and feature
warping; only the learned flow, feature extractor and synthesis networks are
exported to ONNX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ResShift browser ONNX components")
    parser.add_argument("--source", type=Path, required=True, help="Multi-Input-Resshift-Diffusion-VFI checkout")
    parser.add_argument("--model-dir", type=Path, required=True, help="Downloaded Hugging Face checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination for ONNX graphs and manifest")
    parser.add_argument("--height", type=int, default=128, help="Example export height, divisible by 16")
    parser.add_argument("--width", type=int, default=128, help="Example export width, divisible by 16")
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dimensions(height: int, width: int) -> None:
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("ResShift export height and width must be positive multiples of 16")


def export_graph(torch, module, inputs, path: Path, input_names: list[str], output_names: list[str], dynamic_axes: dict[str, dict[int, str]], opset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval().requires_grad_(False)
    print(f"Exporting {path.name}", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
        )


def spatial_axes(names: Iterable[str]) -> dict[str, dict[int, str]]:
    return {name: {0: "batch", 2: f"{name}_height", 3: f"{name}_width"} for name in names}


def validate_onnx(onnx, path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    print(f"Validated {path.name}", flush=True)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    ensure_dimensions(args.height, args.width)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    os.chdir(source)
    sys.path.insert(0, str(source))

    import onnx
    import torch
    import torch.nn as nn
    from model.hub import MultiInputResShiftHub

    print("Loading native ResShift checkpoint on CPU", flush=True)
    model = MultiInputResShiftHub.from_pretrained(str(model_dir)).float().cpu().eval().requires_grad_(False)

    class FlowWrapper(nn.Module):
        def __init__(self, flow_model):
            super().__init__()
            self.flow_model = flow_model

        def forward(self, first, second):
            flow0to1, flow1to0 = self.flow_model(first, second)
            return flow0to1, flow1to0

    class ExtractorWrapper(nn.Module):
        def __init__(self, extractor):
            super().__init__()
            self.extractor = extractor

        def forward(self, image_with_nedt):
            features = self.extractor(image_with_nedt)
            if len(features) != 4:
                raise RuntimeError(f"Expected four ResShift extractor levels, got {len(features)}")
            return tuple(features)

    class SynthesisWrapper(nn.Module):
        def __init__(self, synthesis):
            super().__init__()
            self.synthesis = synthesis

        def forward(
            self,
            sample,
            warp0_0,
            warp0_1,
            warp0_2,
            warp0_3,
            warp0_4,
            warp1_0,
            warp1_1,
            warp1_2,
            warp1_3,
            warp1_4,
            timestep,
        ):
            return self.synthesis(
                sample,
                [warp0_0, warp0_1, warp0_2, warp0_3, warp0_4],
                [warp1_0, warp1_1, warp1_2, warp1_3, warp1_4],
                timestep,
            )

    first = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    second = torch.zeros_like(first)
    flow_path = output / "flow.onnx"
    export_graph(
        torch,
        FlowWrapper(model.flow_model),
        (first, second),
        flow_path,
        ["first", "second"],
        ["flow0to1_dydx", "flow1to0_dydx"],
        {
            "first": {0: "batch", 2: "height", 3: "width"},
            "second": {0: "batch", 2: "height", 3: "width"},
            "flow0to1_dydx": {0: "batch", 2: "height", 3: "width"},
            "flow1to0_dydx": {0: "batch", 2: "height", 3: "width"},
        },
        args.opset,
    )

    image_with_nedt = torch.zeros(1, 4, args.height, args.width, dtype=torch.float32)
    with torch.inference_mode():
        feature_examples = model.feature_warper.feature_extractor(image_with_nedt)
    extractor_path = output / "extractor.onnx"
    extractor_outputs = [f"feature_{index}" for index in range(len(feature_examples))]
    extractor_axes = {"image_with_nedt": {0: "batch", 2: "height", 3: "width"}}
    extractor_axes.update(spatial_axes(extractor_outputs))
    export_graph(
        torch,
        ExtractorWrapper(model.feature_warper.feature_extractor),
        (image_with_nedt,),
        extractor_path,
        ["image_with_nedt"],
        extractor_outputs,
        extractor_axes,
        args.opset,
    )

    sample = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    warp0 = [torch.zeros(1, 5, args.height, args.width, dtype=torch.float32)]
    warp1 = [torch.zeros_like(warp0[0])]
    for feature in feature_examples:
        channels = feature.shape[1] + 1
        height = feature.shape[2]
        width = feature.shape[3]
        warp0.append(torch.zeros(1, channels, height, width, dtype=torch.float32))
        warp1.append(torch.zeros_like(warp0[-1]))
    timestep = torch.zeros(1, dtype=torch.long)
    synthesis_inputs = ["sample"] + [f"warp0_{index}" for index in range(5)] + [f"warp1_{index}" for index in range(5)] + ["timestep"]
    synthesis_axes: dict[str, dict[int, str]] = {"sample": {0: "batch", 2: "height", 3: "width"}, "timestep": {0: "batch"}, "predicted_x0": {0: "batch", 2: "height", 3: "width"}}
    synthesis_axes.update(spatial_axes([name for name in synthesis_inputs if name.startswith("warp")]))
    synthesis_path = output / "synthesis.onnx"
    export_graph(
        torch,
        SynthesisWrapper(model.synthesis),
        (sample, *warp0, *warp1, timestep),
        synthesis_path,
        synthesis_inputs,
        ["predicted_x0"],
        synthesis_axes,
        args.opset,
    )

    for graph in (flow_path, extractor_path, synthesis_path):
        validate_onnx(onnx, graph)

    assets = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        assets.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    manifest = {
        "format": 1,
        "family": "resshift",
        "source": "VicFonch/Multi-Input-Resshift-Diffusion-VFI",
        "components": {
            "flow": "flow.onnx",
            "extractor": "extractor.onnx",
            "synthesis": "synthesis.onnx",
        },
        "flowLayout": "dy-dx",
        "warping": "browser-native",
        "scheduler": "browser-native",
        "opset": args.opset,
        "exampleShape": [1, 3, args.height, args.width],
        "assets": assets,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
