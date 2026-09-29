#!/usr/bin/env python3
"""Export a user-supplied ProPainter installation as a browser repair window.

This tool intentionally does not download or redistribute ProPainter weights.
The upstream pretrained weights are restricted to non-commercial use. Run this
exporter only with weights you are permitted to use and host.

The browser-native selective repair path uses the same 12-frame, neighbour-4,
reference-stride-2 schedule as webp-animator's native ProPainter worker. Python
control flow is unrolled at export time into one fixed repair-window graph.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ProPainter browser repair window")
    parser.add_argument("--source", type=Path, required=True, help="sczhou/ProPainter checkout")
    parser.add_argument("--raft-checkpoint", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--propainter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--raft-iters", type=int, default=12)
    parser.add_argument("--opset", type=int, default=22)
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
    check = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(check)
    files = [path]
    external = path.with_name(data_name)
    if external.is_file():
        files.append(external)
    return files


def const_int(symbolic_helper, value, name: str) -> int:
    return int(symbolic_helper._get_const(value, "i", name))


def register_deform_conv_symbolic(torch, symbolic_helper, opset: int) -> None:
    if opset < 22:
        raise ValueError("ProPainter export requires ONNX opset 22+ for standard DeformConv")

    def deform_conv_symbolic(
        graph,
        input_tensor,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        offset_groups,
        use_mask,
    ):
        stride_h_i = const_int(symbolic_helper, stride_h, "stride_h")
        stride_w_i = const_int(symbolic_helper, stride_w, "stride_w")
        pad_h_i = const_int(symbolic_helper, pad_h, "pad_h")
        pad_w_i = const_int(symbolic_helper, pad_w, "pad_w")
        dilation_h_i = const_int(symbolic_helper, dilation_h, "dilation_h")
        dilation_w_i = const_int(symbolic_helper, dilation_w, "dilation_w")
        groups_i = const_int(symbolic_helper, groups, "groups")
        offset_groups_i = const_int(symbolic_helper, offset_groups, "offset_groups")
        use_mask_i = bool(const_int(symbolic_helper, use_mask, "use_mask"))
        inputs = [input_tensor, weight, offset, bias]
        if use_mask_i:
            inputs.append(mask)
        return graph.op(
            "DeformConv",
            *inputs,
            strides_i=[stride_h_i, stride_w_i],
            pads_i=[pad_h_i, pad_w_i, pad_h_i, pad_w_i],
            dilations_i=[dilation_h_i, dilation_w_i],
            group_i=groups_i,
            offset_group_i=offset_groups_i,
        )

    torch.onnx.register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv_symbolic, opset)


def reference_indices(mid: int, neighbours: list[int], length: int, stride: int) -> list[int]:
    return [index for index in range(0, length, stride) if index not in neighbours]


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    checkpoints = [args.raft_checkpoint.resolve(), args.flow_checkpoint.resolve(), args.propainter_checkpoint.resolve()]
    if not source.is_dir():
        raise FileNotFoundError(source)
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if args.window_size != 12:
        raise ValueError("The browser parity contract currently requires the native 12-frame ProPainter window")
    if args.overlap <= 0 or args.overlap >= args.window_size:
        raise ValueError("overlap must be positive and smaller than window-size")
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        raise ValueError("height and width must be positive multiples of 8")

    os.chdir(source)
    sys.path.insert(0, str(source))

    import onnx
    import torch
    import torch.nn as nn
    from torch.onnx import symbolic_helper
    from model.modules.flow_comp_raft import RAFT_bi
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from model.propainter import InpaintGenerator

    register_deform_conv_symbolic(torch, symbolic_helper, args.opset)

    raft = RAFT_bi(str(checkpoints[0]), device="cpu").cpu().eval().requires_grad_(False)
    flow_complete = RecurrentFlowCompleteNet(str(checkpoints[1])).cpu().eval().requires_grad_(False)
    propainter = InpaintGenerator(model_path=str(checkpoints[2])).cpu().eval().requires_grad_(False)

    class RepairWindow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raft = raft
            self.flow_complete = flow_complete
            self.propainter = propainter

        def forward(self, frames, masks):
            flows_f, flows_b = self.raft(frames, iters=args.raft_iters)
            predicted, _ = self.flow_complete.forward_bidirect_flow((flows_f, flows_b), masks)
            completed_f, completed_b = self.flow_complete.combine_flow(
                (flows_f, flows_b), predicted, masks
            )

            masked_frames = frames * (1 - masks)
            batch, time, _, height, width = masks.shape
            propagated, updated_mask = self.propainter.img_propagation(
                masked_frames,
                (completed_f, completed_b),
                masks,
                "nearest",
            )
            updated_frames = frames * (1 - masks) + propagated.view(batch, time, 3, height, width) * masks
            updated_masks = updated_mask.view(batch, time, 1, height, width)

            neighbour_stride = 2
            ref_stride = 2
            predictions: list[list[torch.Tensor]] = [[] for _ in range(args.window_size)]
            for middle in range(0, args.window_size, neighbour_stride):
                neighbours = list(range(
                    max(0, middle - neighbour_stride),
                    min(args.window_size, middle + neighbour_stride + 1),
                ))
                references = reference_indices(middle, neighbours, args.window_size, ref_stride)
                selected = neighbours + references
                selected_images = updated_frames[:, selected]
                selected_masks = masks[:, selected]
                selected_updated_masks = updated_masks[:, selected]
                flow_indices = neighbours[:-1]
                selected_flows = (
                    completed_f[:, flow_indices],
                    completed_b[:, flow_indices],
                )
                local_count = len(neighbours)
                local_prediction = self.propainter(
                    selected_images,
                    selected_flows,
                    selected_masks,
                    selected_updated_masks,
                    local_count,
                ).view(batch, local_count, 3, height, width)
                for local_index, frame_index in enumerate(neighbours):
                    predictions[frame_index].append(local_prediction[:, local_index])

            repaired = []
            for frame_index, candidates in enumerate(predictions):
                if not candidates:
                    repaired.append(updated_frames[:, frame_index])
                elif len(candidates) == 1:
                    repaired.append(candidates[0])
                else:
                    repaired.append(torch.stack(candidates, dim=0).mean(dim=0))
            return torch.stack(repaired, dim=1)

    module = RepairWindow().eval()
    frames = torch.zeros(1, args.window_size, 3, args.height, args.width, dtype=torch.float32)
    masks = torch.zeros(1, args.window_size, 1, args.height, args.width, dtype=torch.float32)
    masks[:, 1:-1, :, args.height // 3: 2 * args.height // 3, args.width // 3: 2 * args.width // 3] = 1

    output.mkdir(parents=True, exist_ok=True)
    graph = output / "propainter_repair_window.onnx"
    export_kwargs = dict(
        input_names=["frames", "masks"],
        output_names=["repaired"],
        opset_version=args.opset,
        do_constant_folding=True,
        export_params=True,
    )
    if "use_external_data_format" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["use_external_data_format"] = True

    print("Exporting fixed 12-frame ProPainter repair graph...", flush=True)
    with torch.inference_mode():
        torch.onnx.export(module, (frames, masks), str(graph), **export_kwargs)

    files = externalise(onnx, graph)
    manifest = {
        "format": 1,
        "family": "propainter",
        "source": "sczhou/ProPainter",
        "licence": "Upstream ProPainter pretrained weights are restricted to non-commercial use; see the upstream LICENSE and weight terms.",
        "components": {"repairWindow": graph.name},
        "windowSize": args.window_size,
        "overlap": args.overlap,
        "internalWidth": args.width,
        "internalHeight": args.height,
        "raftIterations": args.raft_iters,
        "neighbourLength": 4,
        "referenceStride": 2,
        "opset": args.opset,
        "assets": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(set(files))
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)
    print("Weights were not copied into the repository. Host the exported files only where their licence permits.", flush=True)


if __name__ == "__main__":
    main()
