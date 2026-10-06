#!/usr/bin/env python3
"""Export AMT-S into browser-oriented ONNX scale variants.

AMT's native low-VRAM worker retries the same network at progressively lower
internal scale factors. ONNX cannot treat that Python float as a portable runtime
control without tracing divergent graphs, so this exporter deliberately emits one
graph per native fallback scale. The browser chooses the highest scale that can
run and keeps padding/unpadding outside the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AMT-S browser ONNX variants")
    parser.add_argument("--source", type=Path, required=True, help="MCG-NKU/AMT checkout")
    parser.add_argument("--config", type=Path, required=True, help="AMT-S.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="amt-s.pth")
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


def next_multiple(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def scale_input_divisor(scale_factor: float) -> int:
    """Return an input multiple whose scaled size remains divisible by 16."""
    scale = Fraction(str(scale_factor)).limit_denominator(64)
    return scale.denominator * 16 // math.gcd(scale.numerator, 16)


def export_graph(torch, module, inputs, path: Path, opset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["first", "second", "timestep"],
        output_names=["interpolated"],
        dynamic_axes={
            "first": {0: "batch", 2: "height", 3: "width"},
            "second": {0: "batch", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "interpolated": {0: "batch", 2: "height", 3: "width"},
        },
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
    if args.height <= 0 or args.width <= 0:
        raise ValueError("AMT export dimensions must be positive")
    for path in (source, config_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.chdir(source)
    sys.path.insert(0, str(source))

    import onnx
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    from utils.build_utils import build_from_cfg

    network_cfg = OmegaConf.load(str(config_path)).network
    model = build_from_cfg(network_cfg)
    checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
    state = checkpoint_data.get("state_dict", checkpoint_data)
    model.load_state_dict(state)
    model = model.cpu().eval().requires_grad_(False)

    class AmtWrapper(nn.Module):
        def __init__(self, network, scale_factor: float):
            super().__init__()
            self.network = network
            self.scale_factor = float(scale_factor)

        def forward(self, first, second, timestep):
            return self.network(
                first,
                second,
                timestep,
                scale_factor=self.scale_factor,
                eval=True,
            )["imgt_pred"]

    variants = (
        ("scale100", 1.0),
        ("scale075", 0.75),
        ("scale050", 0.5),
        ("scale025", 0.25),
    )
    components: dict[str, str] = {}
    scale_manifest = []
    asset_files: list[Path] = []
    for component, scale_factor in variants:
        divisor = scale_input_divisor(scale_factor)
        height = next_multiple(args.height, divisor)
        width = next_multiple(args.width, divisor)
        first = torch.zeros(1, 3, height, width, dtype=torch.float32)
        second = torch.zeros_like(first)
        timestep = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float32)
        graph = output / f"amt_s_{component}.onnx"
        export_graph(torch, AmtWrapper(model, scale_factor), (first, second, timestep), graph, args.opset)
        asset_files.extend(externalise_onnx(onnx, graph))
        components[component] = graph.name
        scale_manifest.append({
            "component": component,
            "scaleFactor": scale_factor,
            "divisor": divisor,
        })

    assets = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(set(asset_files))
    ]
    manifest = {
        "format": 1,
        "family": "amt",
        "source": "MCG-NKU/AMT",
        "licence": "CC-BY-NC-4.0",
        "components": components,
        "scales": scale_manifest,
        "opset": args.opset,
        "assets": assets,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
