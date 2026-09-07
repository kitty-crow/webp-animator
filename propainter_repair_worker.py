#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Some older research torch builds reject this newer allocator option at CUDA init.
_raw_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if _raw_alloc:
    _parts = [item.strip() for item in _raw_alloc.split(",") if item.strip()]
    _parts = [item for item in _parts if not item.lower().startswith("expandable_segments:")]
    if _parts:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_parts)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def progress(current: int, total: int) -> None:
    print(f"PROGRESS {current} {total}", flush=True)


def _alpha_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
    return torch.from_numpy(alpha).to(device=device).unsqueeze(0).unsqueeze(0)


def _dilate(value: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(0, int(radius))
    if radius <= 0:
        return value
    return F.max_pool2d(value, kernel_size=radius * 2 + 1, stride=1, padding=radius)


def _audit_frame(
    previous: Image.Image,
    current: Image.Image,
    following: Image.Image,
    device: torch.device,
):
    width, height = current.size
    radius = max(8, min(48, int(round(max(width, height) * 0.02))))
    local_radius = max(3, radius // 3)

    prev_a = _alpha_tensor(previous, device)
    cur_a = _alpha_tensor(current, device)
    next_a = _alpha_tensor(following, device)

    prev_support = _dilate(prev_a, radius)
    next_support = _dilate(next_a, radius)
    both_support = torch.minimum(prev_support, next_support)
    either_support = torch.maximum(prev_support, next_support)
    local_visible = _dilate(cur_a, local_radius)

    # Conservative missing-fill detector. It only marks transparent/weak pixels
    # immediately connected to visible candidate structure and supported by the
    # temporal neighbours, so ordinary transparent background remains untouched.
    strong_hole = (
        (cur_a < 0.12)
        & (local_visible > 0.42)
        & (both_support > 0.52)
    )
    moving_edge_hole = (
        (cur_a < 0.18)
        & (local_visible > 0.58)
        & (either_support > 0.70)
    )
    weak_fill = (
        (cur_a >= 0.01)
        & (cur_a < 0.34)
        & (local_visible > 0.60)
        & (both_support > 0.68)
    )

    mask = strong_hole | moving_edge_hole | weak_fill
    mask = _dilate(mask.float(), 2) > 0.5

    # Reconstruct alpha deterministically from temporal support. ProPainter then
    # reconstructs the RGB content only inside the exact same repair mask.
    expected_alpha = torch.maximum(
        cur_a,
        torch.minimum(
            either_support,
            torch.maximum(both_support, local_visible * 0.88),
        ),
    ).clamp(0.0, 1.0)

    mask_np = (mask[0, 0].detach().cpu().numpy().astype(np.uint8) * 255)
    alpha_np = np.rint(expected_alpha[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)

    del prev_a, cur_a, next_a, prev_support, next_support, both_support, either_support, local_visible, mask, expected_alpha
    torch.cuda.empty_cache()
    return mask_np, alpha_np


def _bbox_from_masks(masks: dict[int, np.ndarray], width: int, height: int):
    xs: list[int] = []
    ys: list[int] = []
    for mask in masks.values():
        y, x = np.where(mask > 0)
        if x.size:
            xs.extend((int(x.min()), int(x.max())))
            ys.extend((int(y.min()), int(y.max())))
    if not xs:
        return None
    pad = max(48, min(128, int(round(max(width, height) * 0.04))))
    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(width, max(xs) + 1 + pad)
    y1 = min(height, max(ys) + 1 + pad)
    return x0, y0, x1, y1


def _processing_size(width: int, height: int, total_vram: int):
    # ProPainter's own published low-memory modes are used here, but we also cap
    # the crop size proactively for the current 4 GB JASPER GPU.
    gib = total_vram / float(1024**3)
    if gib <= 5.25:
        max_pixels = 384 * 288
        max_side = 384
    elif gib <= 9.5:
        max_pixels = 576 * 432
        max_side = 576
    else:
        max_pixels = 720 * 540
        max_side = 720

    scale = min(1.0, max_side / max(width, height))
    scaled_pixels = width * height * scale * scale
    if scaled_pixels > max_pixels:
        scale *= math.sqrt(max_pixels / scaled_pixels)
    out_w = max(8, int(width * scale))
    out_h = max(8, int(height * scale))
    out_w -= out_w % 8
    out_h -= out_h % 8
    return max(8, out_w), max(8, out_h)


def _run_propainter(source: Path, frames_dir: Path, masks_dir: Path, output_dir: Path, process_size: tuple[int, int]):
    width, height = process_size
    command = [
        sys.executable,
        str(source / "inference_propainter.py"),
        "--video",
        str(frames_dir),
        "--mask",
        str(masks_dir),
        "--output",
        str(output_dir),
        "--width",
        str(width),
        "--height",
        str(height),
        "--neighbor_length",
        "4",
        "--ref_stride",
        "2",
        "--subvideo_length",
        "12",
        "--raft_iter",
        "12",
        "--mask_dilation",
        "2",
        "--save_frames",
        "--fp16",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(source),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    tail: list[str] = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip()
        if line:
            print(line, flush=True)
            tail.append(line)
            tail = tail[-60:]
    code = process.wait()
    if code != 0:
        raise RuntimeError("ProPainter inference failed:\n" + "\n".join(tail[-20:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--propainter-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    source = Path(args.propainter_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    result_path = Path(args.result).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_paths = [Path(path).resolve() for path in manifest.get("frames", [])]
    targets = sorted({int(index) for index in manifest.get("targets", [])})
    output_dir = Path(manifest.get("output_dir") or result_path.parent / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("ProPainter temporal repair requires CUDA; no CUDA GPU is available.")

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    frames = [Image.open(path).convert("RGBA").copy() for path in frame_paths]
    if not frames:
        raise RuntimeError("Temporal repair received no frames.")
    size = frames[0].size
    if any(frame.size != size for frame in frames):
        raise RuntimeError("Temporal repair requires a common frame canvas.")
    width, height = size

    progress(1, 3)
    masks: dict[int, np.ndarray] = {}
    expected_alpha: dict[int, np.ndarray] = {}
    for index in targets:
        if index < 0 or index >= len(frames):
            continue
        previous = frames[(index - 1) % len(frames)]
        following = frames[(index + 1) % len(frames)]
        mask, alpha = _audit_frame(previous, frames[index], following, device)
        if int(np.count_nonzero(mask)) < 12:
            continue
        masks[index] = mask
        expected_alpha[index] = alpha

    mask_pixels = int(sum(np.count_nonzero(mask) for mask in masks.values()))
    bbox = _bbox_from_masks(masks, width, height)
    if bbox is None:
        result_path.write_text(
            json.dumps(
                {
                    "frames": [],
                    "repaired": 0,
                    "mask_pixels": 0,
                    "cuda": True,
                    "device": torch.cuda.get_device_name(0),
                    "processing_size": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        progress(3, 3)
        return

    x0, y0, x1, y1 = bbox
    crop_w, crop_h = x1 - x0, y1 - y0
    process_size = _processing_size(
        crop_w,
        crop_h,
        int(torch.cuda.get_device_properties(0).total_memory),
    )

    work = output_dir / "work"
    shutil.rmtree(work, ignore_errors=True)
    input_frames = work / "frames"
    input_masks = work / "masks"
    propainter_output = work / "propainter"
    input_frames.mkdir(parents=True, exist_ok=True)
    input_masks.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(frames):
        crop = frame.crop((x0, y0, x1, y1))
        # ProPainter is RGB-only. Fully transparent source pixels are deliberately
        # presented as black, but only model pixels inside our repair mask are ever
        # copied back into the RGBA result.
        crop.convert("RGB").save(input_frames / f"{index:04d}.png")
        if index in masks:
            mask_crop = masks[index][y0:y1, x0:x1]
        else:
            mask_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
        Image.fromarray(mask_crop, mode="L").save(input_masks / f"{index:04d}.png")

    progress(2, 3)
    _run_propainter(source, input_frames, input_masks, propainter_output, process_size)

    repaired_dir = output_dir / "frames"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    pp_frames = propainter_output / input_frames.name / "frames"
    returned: list[dict] = []
    repaired_count = 0

    for index in targets:
        if index not in masks:
            continue
        pp_path = pp_frames / f"{index:04d}.png"
        if not pp_path.is_file():
            raise RuntimeError(f"ProPainter did not return repaired frame {index}.")

        original = np.array(frames[index].convert("RGBA"), dtype=np.uint8)
        model_rgb = np.array(Image.open(pp_path).convert("RGB"), dtype=np.uint8)
        if model_rgb.shape[1] != crop_w or model_rgb.shape[0] != crop_h:
            model_rgb = np.array(
                Image.fromarray(model_rgb, mode="RGB").resize((crop_w, crop_h), Image.Resampling.BICUBIC),
                dtype=np.uint8,
            )

        mask_crop = masks[index][y0:y1, x0:x1] > 0
        region = original[y0:y1, x0:x1]
        region_rgb = region[..., :3]
        region_rgb[mask_crop] = model_rgb[mask_crop]
        region[..., :3] = region_rgb
        alpha_crop = expected_alpha[index][y0:y1, x0:x1]
        region_alpha = region[..., 3]
        region_alpha[mask_crop] = alpha_crop[mask_crop]
        region[..., 3] = region_alpha
        original[y0:y1, x0:x1] = region

        repaired_path = repaired_dir / f"{index:06d}.png"
        Image.fromarray(original, mode="RGBA").save(repaired_path)
        returned.append({"index": index, "path": str(repaired_path), "mask_pixels": int(np.count_nonzero(masks[index]))})
        repaired_count += 1

    result_path.write_text(
        json.dumps(
            {
                "frames": returned,
                "repaired": repaired_count,
                "mask_pixels": mask_pixels,
                "cuda": True,
                "device": torch.cuda.get_device_name(0),
                "processing_size": [process_size[0], process_size[1]],
                "crop": [x0, y0, x1, y1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    progress(3, 3)


if __name__ == "__main__":
    main()
