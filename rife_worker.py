#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Reduce CUDA allocator fragmentation when supported. This must be set before
# Practical-RIFE imports torch. Callers can still override it explicitly.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

RIFE_SCALES = (1.0, 0.5, 0.25)


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


def _padding_multiple(scale: float) -> int:
    # Match Practical-RIFE's own inference_video.py padding rule. Lower internal
    # scales need a larger input multiple so every IFNet stage remains aligned.
    return max(128, int(128 / scale))


def tensor_from_rgb(torch, image: Image.Image, device, scale: float):
    from torch.nn import functional as F

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    h, w = array.shape[:2]
    multiple = _padding_multiple(scale)
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple
    return F.pad(tensor, (0, pw - w, 0, ph - h)), h, w


def tensor_from_alpha(torch, image: Image.Image, device, scale: float):
    from torch.nn import functional as F

    alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
    array = np.repeat(alpha[..., None], 3, axis=2)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    h, w = alpha.shape
    multiple = _padding_multiple(scale)
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple
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


def _clear_cuda_after_oom(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _infer_with_scale_fallback(
    torch,
    model,
    first: Image.Image,
    second: Image.Image,
    ratio: float,
    *,
    alpha: bool,
):
    """Run one RIFE inference, reducing internal scale only after CUDA OOM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    make_tensor = tensor_from_alpha if alpha else tensor_from_rgb
    convert = alpha_from_tensor if alpha else rgb_from_tensor
    last_error = None

    # CPU does not need a VRAM fallback. Keep full internal resolution there.
    scales = RIFE_SCALES if device.type == "cuda" else (1.0,)

    for index, scale in enumerate(scales):
        first_tensor = None
        second_tensor = None
        middle = None
        try:
            first_tensor, h, w = make_tensor(torch, first, device, scale)
            second_tensor, _, _ = make_tensor(torch, second, device, scale)
            with torch.inference_mode():
                middle = model.inference(first_tensor, second_tensor, ratio, scale)
            result = convert(middle, h, w)
            return result, scale
        except torch.OutOfMemoryError as exc:
            last_error = exc
            # Drop every tensor from the failed attempt before asking the CUDA
            # allocator for a smaller inference. PyTorch's model stays resident.
            del middle, second_tensor, first_tensor
            _clear_cuda_after_oom(torch)
            if index + 1 < len(scales):
                next_scale = scales[index + 1]
                channel = "alpha" if alpha else "RGB"
                print(
                    f"RIFE CUDA OOM for {channel} at scale={scale:g}; "
                    f"retrying at scale={next_scale:g}",
                    file=sys.stderr,
                    flush=True,
                )

    if last_error is not None:
        raise last_error
    raise RuntimeError("RIFE inference failed without an error.")


def interpolate_pair(torch, model, first: Image.Image, second: Image.Image, ratio: float):
    first = first.convert("RGBA")
    second = second.convert("RGBA")
    if first.size != second.size:
        raise ValueError("RIFE input frames must have the same canvas size.")

    rgb, rgb_scale = _infer_with_scale_fallback(
        torch, model, first, second, ratio, alpha=False
    )

    first_alpha = np.asarray(first.getchannel("A"))
    second_alpha = np.asarray(second.getchannel("A"))
    opaque = np.all(first_alpha == 255) and np.all(second_alpha == 255)

    if opaque:
        alpha = np.full((first.height, first.width), 255, dtype=np.uint8)
        alpha_scale = rgb_scale
    else:
        alpha, alpha_scale = _infer_with_scale_fallback(
            torch, model, first, second, ratio, alpha=True
        )

    if rgb_scale != 1.0 or alpha_scale != 1.0:
        print(
            f"RIFE_SCALE rgb={rgb_scale:g} alpha={alpha_scale:g}",
            file=sys.stderr,
            flush=True,
        )

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
