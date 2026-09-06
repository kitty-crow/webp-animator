#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from worker_frame_common import numeric_pngs


def parse_args():
    parser = argparse.ArgumentParser(description="AMT interpolation worker")
    parser.add_argument("--amt-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--multi", type=int, choices=(2, 4, 8), required=True)
    return parser.parse_args()


def _run_demo(amt_dir: Path, checkpoint: Path, inputs: list[Path], output: Path, iterations: int):
    command = [
        sys.executable,
        str(amt_dir / "demos" / "demo_2x.py"),
        "-c", str(amt_dir / "cfgs" / "AMT-S.yaml"),
        "-p", str(checkpoint),
        "-n", str(iterations),
        "-o", str(output),
        "--save_images",
        "-i", *map(str, inputs),
    ]
    subprocess.run(command, cwd=str(amt_dir), check=True)
    samples = sorted((output / "samples").glob("sample_*.png"))
    if not samples:
        raise RuntimeError("AMT demo produced no image samples.")
    return samples


def main():
    args = parse_args()
    input_paths = numeric_pngs(args.input_dir.resolve())
    if not input_paths:
        raise SystemExit("At least one aligned PNG frame is required for AMT.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        Image.open(input_paths[0]).convert("RGBA").save(args.output_dir / "000000.png")
        print("PROGRESS 1 1", flush=True)
        return

    iterations = {2: 1, 4: 2, 8: 3}[args.multi]
    with tempfile.TemporaryDirectory(prefix="amt_worker_") as temp_name:
        temp = Path(temp_name)
        rgb_in = temp / "rgb_in"
        alpha_in = temp / "alpha_in"
        rgb_out = temp / "rgb_out"
        alpha_out = temp / "alpha_out"
        rgb_in.mkdir(); alpha_in.mkdir()
        has_alpha = False

        for index, path in enumerate(input_paths):
            rgba = Image.open(path).convert("RGBA")
            rgba.convert("RGB").save(rgb_in / f"{index:06d}.png")
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
            if not np.all(alpha == 255):
                has_alpha = True
            Image.fromarray(np.repeat(alpha[..., None], 3, axis=2), "RGB").save(alpha_in / f"{index:06d}.png")

        rgb_samples = _run_demo(args.amt_dir.resolve(), args.checkpoint.resolve(), numeric_pngs(rgb_in), rgb_out, iterations)
        print("PROGRESS 1 2", flush=True)
        if has_alpha:
            alpha_samples = _run_demo(args.amt_dir.resolve(), args.checkpoint.resolve(), numeric_pngs(alpha_in), alpha_out, iterations)
        else:
            alpha_samples = []

        for index, rgb_path in enumerate(rgb_samples):
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
            if has_alpha:
                alpha_rgb = np.asarray(Image.open(alpha_samples[index]).convert("RGB"), dtype=np.uint8)
                alpha = np.clip(alpha_rgb.mean(axis=2), 0, 255).astype(np.uint8)
            else:
                alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            Image.fromarray(np.dstack((rgb, alpha)), "RGBA").save(args.output_dir / f"{index:06d}.png")
        print("PROGRESS 2 2", flush=True)


if __name__ == "__main__":
    main()
