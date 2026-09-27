#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import engine_paths
import setup_amt
import setup_eden
import setup_engine_common
import setup_mog
import setup_propainter
import setup_resshift
import setup_rife
import setup_speed
import setup_tooncrafter
import temporal_repair


ROOT = Path(__file__).resolve().parent


def _run_check(name: str, callback, results: list[tuple[str, str, str]]) -> None:
    print(f"\n{'=' * 72}\n{name} preflight\n{'=' * 72}")
    try:
        callback()
    except subprocess.CalledProcessError as exc:
        message = f"subprocess exited with code {exc.returncode}"
        results.append((name, "FAIL", message))
        print(f"{name}: FAIL · {message}")
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        results.append((name, "FAIL", message))
        print(f"{name}: FAIL · {message}")
    else:
        results.append((name, "PASS", "runtime preflight passed"))
        print(f"{name}: PASS")


def main() -> int:
    results: list[tuple[str, str, str]] = []

    import app as legacy
    import rife_compat

    rife_compat.install(legacy)
    ready, python, source, model = legacy.rife_paths()
    if ready:
        _run_check("RIFE", lambda: setup_rife.validate_runtime(Path(python)), results)
    else:
        results.append(("RIFE", "SKIP", "not installed"))

    ready, python, source, config, checkpoint = engine_paths.amt_paths()
    if ready:
        _run_check("AMT", lambda: setup_amt.validate_runtime(Path(python)), results)
    else:
        results.append(("AMT", "SKIP", "not installed"))

    ready, python, source, config, checkpoint = engine_paths.eden_paths()
    if ready:
        _run_check("EDEN", lambda: setup_eden.validate_runtime(Path(python)), results)
    else:
        results.append(("EDEN", "SKIP", "not installed"))

    ready, python, source, config, checkpoint = engine_paths.speed_paths()
    if ready:
        _run_check("SPEED", lambda: setup_speed.validate_runtime(Path(python)), results)
    else:
        results.append(("SPEED", "SKIP", "not installed"))

    ready, python, source, model = engine_paths.resshift_paths()
    if ready:
        def resshift_check():
            runtime = setup_engine_common.installed_torch_cuda(Path(python))
            if not runtime:
                raise RuntimeError("ResShift PyTorch does not report a CUDA runtime.")
            setup_engine_common.validate_cuda_runtime(Path(python), runtime, require_cupy=True)
            setup_resshift._validate_runtime(Path(python))
        _run_check("ResShift", resshift_check, results)
    else:
        results.append(("ResShift", "SKIP", "not installed"))

    mog_variants: list[tuple[str, Path]] = []
    mog_python: Path | None = None
    mog_flow: Path | None = None
    for variant, label in (("ani", "MoG animation"), ("real", "MoG real-world")):
        ready, python, source, config, checkpoint, flow_checkpoint = engine_paths.mog_paths(variant)
        if ready:
            mog_python = Path(python)
            mog_flow = Path(flow_checkpoint)
            mog_variants.append((label, Path(checkpoint)))
        else:
            results.append((label, "SKIP", "not installed"))
    if mog_variants and mog_python is not None and mog_flow is not None:
        labels = " + ".join(label for label, _ in mog_variants)
        _run_check(
            labels,
            lambda: setup_mog.validate_runtime(
                mog_python,
                [checkpoint for _, checkpoint in mog_variants],
                mog_flow,
            ),
            results,
        )

    ready, python, source, config, checkpoint = engine_paths.tooncrafter_paths()
    if ready:
        _run_check(
            "ToonCrafter",
            lambda: setup_tooncrafter.validate_runtime(Path(python), Path(checkpoint)),
            results,
        )
    else:
        results.append(("ToonCrafter", "SKIP", "not installed"))

    ready, python, source = temporal_repair.propainter_paths()
    if ready:
        _run_check("ProPainter", lambda: setup_propainter.validate_runtime(Path(python)), results)
    else:
        results.append(("ProPainter", "SKIP", "not installed"))

    print(f"\n{'=' * 72}\nENGINE PREFLIGHT SUMMARY\n{'=' * 72}")
    width = max((len(name) for name, _, _ in results), default=8)
    for name, status, detail in results:
        print(f"{name:<{width}}  {status:<4}  {detail}")

    failures = [item for item in results if item[1] == "FAIL"]
    if failures:
        print(f"\n{len(failures)} installed engine preflight(s) failed.")
        return 1
    print("\nAll installed engine preflights passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
