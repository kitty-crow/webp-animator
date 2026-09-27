#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate every installed WebP Animator engine before using the UI."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Also instantiate the very large MoG/ToonCrafter models and prove their "
            "diffusion cores fit on the current GPU. This can take several minutes."
        ),
    )
    return parser.parse_args()


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


def _cuda_check(python: Path, *, cupy: bool = False) -> None:
    runtime = setup_engine_common.installed_torch_cuda(python)
    if not runtime:
        raise RuntimeError(f"{python} does not report a PyTorch CUDA runtime.")
    setup_engine_common.validate_cuda_runtime(python, runtime, require_cupy=cupy)


def _optional_xformers_check(python: Path) -> None:
    """If xFormers is installed, prove that its attention op supports this GPU."""
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.util, sys; "
                "spec=importlib.util.find_spec('xformers'); "
                "sys.exit(0) if spec is None else None; "
                "import torch, xformers.ops; "
                "assert torch.cuda.is_available(); "
                "q=torch.randn((2,16,32),device='cuda',dtype=torch.float16); "
                "y=xformers.ops.memory_efficient_attention(q,q,q); "
                "torch.cuda.synchronize(); "
                "print('xFormers attention kernel: ok', tuple(y.shape))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "xFormers kernel probe failed").strip()
        raise RuntimeError(
            "xFormers is installed but its memory-efficient-attention kernel is not usable on this GPU. "
            "Rerun that engine's setup so it removes xFormers and selects the PyTorch fallback.\n"
            + details[-1200:]
        )
    if result.stdout.strip():
        print(result.stdout.strip())


def _full_mog_check(
    python: Path,
    source: Path,
    config: Path,
    checkpoint: Path,
    flow_checkpoint: Path,
    variant: str,
) -> None:
    code = f"""
import gc
import sys
from pathlib import Path
root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
import mog_selective_worker as entry
impl = entry.implementation
torch, device, model, inference, total_vram = impl.load_model(
    Path({str(source)!r}),
    Path({str(config)!r}),
    Path({str(checkpoint)!r}),
    Path({str(flow_checkpoint)!r}),
)
torch.cuda.synchronize()
print('MoG {variant} full model GPU placement: ok', f'{{total_vram / 1024**3:.1f}}GB GPU')
del model, inference
gc.collect()
torch.cuda.empty_cache()
"""
    setup_engine_common.run([python, "-c", code])


def _full_tooncrafter_check(
    python: Path,
    source: Path,
    config: Path,
    checkpoint: Path,
) -> None:
    code = f"""
import gc
import sys
from pathlib import Path
root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
import tooncrafter_selective_worker_entry as entry
impl = entry.implementation
torch, device, model, total_vram = impl.load_model(
    Path({str(source)!r}),
    Path({str(config)!r}),
    Path({str(checkpoint)!r}),
)
impl._move_core(model, device)
torch.cuda.synchronize()
print('ToonCrafter diffusion-core GPU placement: ok', f'{{total_vram / 1024**3:.1f}}GB GPU')
impl._move_core(model, torch.device('cpu'))
del model
gc.collect()
torch.cuda.empty_cache()
"""
    setup_engine_common.run([python, "-c", code])


def main() -> int:
    args = parse_args()
    results: list[tuple[str, str, str]] = []

    import app as legacy
    import rife_compat

    rife_compat.install(legacy)
    ready, python, source, model = legacy.rife_paths()
    if ready:
        def rife_check():
            _cuda_check(Path(python))
            setup_rife.validate_runtime(Path(python))
        _run_check("RIFE", rife_check, results)
    else:
        results.append(("RIFE", "SKIP", "not installed"))

    ready, python, source, config, checkpoint = engine_paths.amt_paths()
    if ready:
        def amt_check():
            _cuda_check(Path(python))
            setup_amt.validate_runtime(Path(python))
        _run_check("AMT", amt_check, results)
    else:
        results.append(("AMT", "SKIP", "not installed"))

    ready, python, source, config, checkpoint = engine_paths.eden_paths()
    if ready:
        def eden_check():
            _cuda_check(Path(python), cupy=True)
            _optional_xformers_check(Path(python))
            setup_eden.validate_runtime(Path(python))
        _run_check("EDEN", eden_check, results)
    else:
        results.append(("EDEN", "SKIP", "not installed"))

    ready, python, source, config, checkpoint = engine_paths.speed_paths()
    if ready:
        def speed_check():
            _cuda_check(Path(python))
            setup_speed.validate_runtime(Path(python))
        _run_check("SPEED", speed_check, results)
    else:
        results.append(("SPEED", "SKIP", "not installed"))

    ready, python, source, model = engine_paths.resshift_paths()
    if ready:
        def resshift_check():
            _cuda_check(Path(python), cupy=True)
            setup_resshift._validate_runtime(Path(python))
        _run_check("ResShift", resshift_check, results)
    else:
        results.append(("ResShift", "SKIP", "not installed"))

    mog_variants: list[tuple[str, str, Path, Path]] = []
    mog_python: Path | None = None
    mog_source: Path | None = None
    mog_flow: Path | None = None
    for variant, label in (("ani", "MoG animation"), ("real", "MoG real-world")):
        ready, python, source, config, checkpoint, flow_checkpoint = engine_paths.mog_paths(variant)
        if ready:
            mog_python = Path(python)
            mog_source = Path(source)
            mog_flow = Path(flow_checkpoint)
            mog_variants.append((variant, label, Path(config), Path(checkpoint)))
        else:
            results.append((label, "SKIP", "not installed"))
    if mog_variants and mog_python is not None and mog_source is not None and mog_flow is not None:
        labels = " + ".join(label for _, label, _, _ in mog_variants)
        def mog_check():
            _cuda_check(mog_python, cupy=True)
            _optional_xformers_check(mog_python)
            setup_mog.validate_runtime(
                mog_python,
                [checkpoint for _, _, _, checkpoint in mog_variants],
                mog_flow,
            )
            if args.full:
                for variant, _, config, checkpoint in mog_variants:
                    _full_mog_check(
                        mog_python,
                        mog_source,
                        config,
                        checkpoint,
                        mog_flow,
                        variant,
                    )
        _run_check(labels, mog_check, results)

    ready, python, source, config, checkpoint = engine_paths.tooncrafter_paths()
    if ready:
        def toon_check():
            _cuda_check(Path(python))
            _optional_xformers_check(Path(python))
            setup_tooncrafter.validate_runtime(Path(python), Path(checkpoint))
            if args.full:
                _full_tooncrafter_check(
                    Path(python),
                    Path(source),
                    Path(config),
                    Path(checkpoint),
                )
        _run_check("ToonCrafter", toon_check, results)
    else:
        results.append(("ToonCrafter", "SKIP", "not installed"))

    ready, python, source = temporal_repair.propainter_paths()
    if ready:
        def propainter_check():
            _cuda_check(Path(python))
            setup_propainter.validate_runtime(Path(python))
        _run_check("ProPainter", propainter_check, results)
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
    if not args.full:
        print("Run again with --full to qualify MoG/ToonCrafter full model placement on this GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
