#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Practical-RIFE frame interpolation worker")
    parser.add_argument("--rife-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--multi", type=int, choices=(2, 4, 8), required=True)
    return parser.parse_args()


def load_model(rife_dir: Path, model_dir: Path):
    os.chdir(rife_dir)
    sys.path.insert(0, str(rife_dir))

    import torch

    candidates = []
    try:
        from model.RIFE_HDv2 import Model as ModelV2
        candidates.append(ModelV2)
    except Exception:
        pass
    try:
        from train_log.RIFE_HDv3 import Model as ModelV3
        candidates.append(ModelV3)
    except Exception:
        pass
    try:
        from model.RIFE_HD import Model as ModelV1
        candidates.append(ModelV1)
    except Exception:
        pass

    last_error = None
    model = None
    for ModelClass in candidates:
        try:
            candidate = ModelClass()
            candidate.load_model(str(model_dir), -1)
            model = candidate
            break
        except Exception as exc:
            last_error = exc

    if model is None:
        raise RuntimeError(f"Could not load the RIFE model: {last_error}")

    if not hasattr(model, "version"):
        model.version = 0
    model.eval()
    model.device()

    if model.version < 3.9:
        raise RuntimeError(
            "This integration expects a modern Practical-RIFE model with direct "
            "ratio inference (v3.9+). Install the recommended RIFE 4.25 model."
        )

    return torch, model


def tensor_from_rgb(torch, image: Image.Image, device):
    from torch.nn import functional as F

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    h, w = array.shape[:2]
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    return F.pad(tensor, (0, pw - w, 0, ph - h)), h, w


def tensor_from_alpha(torch, image: Image.Image, device):
    from torch.nn import functional as F

    alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
    array = np.repeat(alpha[..., None], 3, axis=2)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    h, w = alpha.shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    return F.pad(tensor, (0, pw - w, 0, ph - h)), h, w


def rgb_from_tensor(tensor, h: int, w: int):
    array = (
        tensor[0, :, :h, :w]
        .clamp(0, 1)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    return array


def alpha_from_tensor(tensor, h: int, w: int):
    array = (
        tensor[0, 0, :h, :w]
        .clamp(0, 1)
        .mul(255)
        .byte()
        .cpu()
        .numpy()
    )
    return array


def interpolate_pair(torch, model, first: Image.Image, second: Image.Image, ratio: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first = first.convert("RGBA")
    second = second.convert("RGBA")
    if first.size != second.size:
        raise ValueError("RIFE input frames must have the same canvas size.")

    first_rgb, h, w = tensor_from_rgb(torch, first, device)
    second_rgb, _, _ = tensor_from_rgb(torch, second, device)

    with torch.inference_mode():
        middle_rgb = model.inference(first_rgb, second_rgb, ratio)

    rgb = rgb_from_tensor(middle_rgb, h, w)

    first_alpha = np.asarray(first.getchannel("A"))
    second_alpha = np.asarray(second.getchannel("A"))
    opaque = np.all(first_alpha == 255) and np.all(second_alpha == 255)

    if opaque:
        alpha = np.full((h, w), 255, dtype=np.uint8)
    else:
        alpha0, _, _ = tensor_from_alpha(torch, first, device)
        alpha1, _, _ = tensor_from_alpha(torch, second, device)
        with torch.inference_mode():
            middle_alpha = model.inference(alpha0, alpha1, ratio)
        alpha = alpha_from_tensor(middle_alpha, h, w)

    rgba = np.dstack((rgb, alpha))
    return Image.fromarray(rgba, "RGBA")


def numeric_pngs(directory: Path):
    return sorted(directory.glob("*.png"), key=lambda path: int(path.stem))


def main():
    args = parse_args()

    # Practical-RIFE changes the process working directory while loading the model.
    # Resolve every caller-supplied path first so relative Windows paths remain valid
    # after that chdir (and for the same reason on Linux/macOS).
    rife_dir = args.rife_dir.resolve()
    model_dir = args.model_dir.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = numeric_pngs(input_dir)
    if len(paths) < 2:
        raise SystemExit("At least two aligned PNG frames are required for RIFE.")

    torch, model = load_model(rife_dir, model_dir)
    total = (len(paths) - 1) * (args.multi - 1)
    completed = 0
    output_index = 0

    first = Image.open(paths[0]).convert("RGBA")
    first.save(output_dir / f"{output_index:06d}.png")
    output_index += 1

    for pair_index in range(len(paths) - 1):
        frame0 = Image.open(paths[pair_index]).convert("RGBA")
        frame1 = Image.open(paths[pair_index + 1]).convert("RGBA")

        for step in range(1, args.multi):
            ratio = step / args.multi
            middle = interpolate_pair(torch, model, frame0, frame1, ratio)
            middle.save(output_dir / f"{output_index:06d}.png")
            output_index += 1
            completed += 1
            print(f"PROGRESS {completed} {total}", flush=True)

        frame1.save(output_dir / f"{output_index:06d}.png")
        output_index += 1


if __name__ == "__main__":
    main()
