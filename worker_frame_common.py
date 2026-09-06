#!/usr/bin/env python3
from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageChops

CONTENT_MARGIN = 64


def numeric_pngs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.png"), key=lambda path: int(path.stem))


def release_cuda(torch_module=None) -> None:
    gc.collect()
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _content_bbox(first: Image.Image, second: Image.Image):
    union = ImageChops.lighter(first.getchannel("A"), second.getchannel("A"))
    bbox = union.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return (
        max(0, left - CONTENT_MARGIN),
        max(0, top - CONTENT_MARGIN),
        min(first.width, right + CONTENT_MARGIN),
        min(first.height, bottom + CONTENT_MARGIN),
    )


def generated_midpoint(
    first: Image.Image,
    second: Image.Image,
    infer_rgb: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Image.Image:
    first = first.convert("RGBA")
    second = second.convert("RGBA")
    if first.size != second.size:
        raise ValueError("Generator endpoints must have the same canvas size.")

    bbox = _content_bbox(first, second)
    if bbox is None:
        return Image.new("RGBA", first.size, (0, 0, 0, 0))

    first_crop = first.crop(bbox)
    second_crop = second.crop(bbox)
    rgb0 = np.asarray(first_crop.convert("RGB"), dtype=np.uint8)
    rgb1 = np.asarray(second_crop.convert("RGB"), dtype=np.uint8)
    rgb_mid = np.asarray(infer_rgb(rgb0, rgb1), dtype=np.uint8)

    alpha0 = np.asarray(first_crop.getchannel("A"), dtype=np.uint8)
    alpha1 = np.asarray(second_crop.getchannel("A"), dtype=np.uint8)
    if np.all(alpha0 == 255) and np.all(alpha1 == 255):
        alpha_mid = np.full(alpha0.shape, 255, dtype=np.uint8)
    else:
        mask0 = np.repeat(alpha0[..., None], 3, axis=2)
        mask1 = np.repeat(alpha1[..., None], 3, axis=2)
        generated_mask = np.asarray(infer_rgb(mask0, mask1), dtype=np.float32)
        alpha_mid = np.clip(generated_mask.mean(axis=2), 0, 255).astype(np.uint8)

    region = Image.fromarray(np.dstack((rgb_mid, alpha_mid)), "RGBA")
    output = Image.new("RGBA", first.size, (0, 0, 0, 0))
    output.paste(region, (bbox[0], bbox[1]))
    return output


def write_generated_sequence(
    input_dir: Path,
    output_dir: Path,
    infer_rgb: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    paths = numeric_pngs(input_dir)
    if not paths:
        raise SystemExit("At least one aligned PNG frame is required.")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_index = 0
    first = Image.open(paths[0]).convert("RGBA")
    first.save(output_dir / f"{output_index:06d}.png")
    output_index += 1

    if len(paths) == 1:
        print("PROGRESS 1 1", flush=True)
        return

    total = len(paths) - 1
    for index in range(total):
        frame0 = Image.open(paths[index]).convert("RGBA")
        frame1 = Image.open(paths[index + 1]).convert("RGBA")
        midpoint = generated_midpoint(frame0, frame1, infer_rgb)
        midpoint.save(output_dir / f"{output_index:06d}.png")
        output_index += 1
        frame1.save(output_dir / f"{output_index:06d}.png")
        output_index += 1
        print(f"PROGRESS {index + 1} {total}", flush=True)
