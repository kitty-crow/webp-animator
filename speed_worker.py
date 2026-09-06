#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from worker_frame_common import release_cuda, write_generated_sequence


def parse_args():
    parser = argparse.ArgumentParser(description="SPEED midpoint generator worker")
    parser.add_argument("--speed-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    return parser.parse_args()


def _load_inference_module(speed_dir: Path):
    sys.path.insert(0, str(speed_dir))
    spec = importlib.util.spec_from_file_location("speed_upstream_inference", speed_dir / "inference.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load SPEED inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pad_to_multiple(array: np.ndarray, multiple: int = 64):
    h, w = array.shape[:2]
    ph = ((h + multiple - 1) // multiple) * multiple
    pw = ((w + multiple - 1) // multiple) * multiple
    if (ph, pw) == (h, w):
        return array, h, w
    return np.pad(array, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge"), h, w


def main():
    args = parse_args()
    speed_dir = args.speed_dir.resolve()
    module = _load_inference_module(speed_dir)
    torch = module.require_torch()
    device = module.get_device("cuda:0")
    model = module.build_model(
        str(speed_dir / "configs" / "eval_config.yaml"),
        str(args.checkpoint.resolve()),
        device,
        strict_load=False,
    )

    def infer_rgb(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        original_h, original_w = left.shape[:2]
        candidates = [None, 768, 512, 384]
        last_error = None
        for max_side in candidates:
            if max_side is None or max(original_h, original_w) <= max_side:
                l_img = Image.fromarray(left, "RGB")
                r_img = Image.fromarray(right, "RGB")
            else:
                scale = max_side / max(original_h, original_w)
                size = (max(1, round(original_w * scale)), max(1, round(original_h * scale)))
                l_img = Image.fromarray(left, "RGB").resize(size, Image.Resampling.LANCZOS)
                r_img = Image.fromarray(right, "RGB").resize(size, Image.Resampling.LANCZOS)
            l_arr, h, w = _pad_to_multiple(np.asarray(l_img))
            r_arr, _, _ = _pad_to_multiple(np.asarray(r_img))
            try:
                pred = module.interpolate_batch(model, [l_arr], [r_arr], device, precision=args.precision)[0][:h, :w]
                if (w, h) != (original_w, original_h):
                    pred = np.asarray(Image.fromarray(pred, "RGB").resize((original_w, original_h), Image.Resampling.LANCZOS))
                release_cuda(torch)
                return pred
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                last_error = exc
                release_cuda(torch)
                continue
        if last_error:
            raise last_error
        raise RuntimeError("SPEED inference failed.")

    write_generated_sequence(args.input_dir.resolve(), args.output_dir.resolve(), infer_rgb)
    del model
    release_cuda(torch)


if __name__ == "__main__":
    main()
