#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

from worker_common import load_rgba, read_manifest, recursive_midpoints, write_result


def parse_args():
    parser = argparse.ArgumentParser(description="Selective Practical-RIFE interpolation worker")
    parser.add_argument("--rife-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = read_manifest(args.manifest)
    output_dir = Path(manifest["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from rife_worker import interpolate_pair, load_model, _release_cuda

    torch, model = load_model(args.rife_dir.resolve(), args.model_dir.resolve())
    tasks = list(manifest.get("tasks", []))
    completed = 0
    total_expected = sum(max(0, (2 ** int(task.get("depth", 1))) - 1) for task in tasks)
    result_tasks = []

    for task_index, task in enumerate(tasks):
        first = load_rgba(Path(task["left"]))
        second = load_rgba(Path(task["right"]))
        counter = 0

        def generate(a, b):
            return interpolate_pair(torch, model, a, b, 0.5)

        def save(image, t):
            nonlocal counter, completed
            path = output_dir / f"{task_index:04d}_{counter:04d}.png"
            counter += 1
            image.save(path)
            completed += 1
            print(f"PROGRESS {completed} {max(1, total_expected)}", flush=True)
            return {"path": str(path), "t": float(t)}

        threshold = task.get("threshold")
        frames = recursive_midpoints(
            first,
            second,
            depth=max(0, int(task.get("depth", 1))),
            threshold=None if threshold is None else float(threshold),
            generate_midpoint=generate,
            save_midpoint=save,
            alpha_threshold=int(task.get("alpha_threshold", 8)),
        )
        result_tasks.append({"id": task.get("id", str(task_index)), "frames": frames})
        _release_cuda(torch)
        gc.collect()

    write_result(args.result, {"tasks": result_tasks})


if __name__ == "__main__":
    main()
