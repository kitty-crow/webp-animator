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
        dest="full",
        action="store_true",
        help=(
            "Run real one-step inference through MoG/ToonCrafter, including RAM/disk "
            "offload hooks. This is the default and is the release-qualification path."
        ),
    )
    parser.add_argument(
        "--quick",
        dest="full",
        action="store_false",
        help=(
            "Skip expensive MoG/ToonCrafter inference and only check environments, "
            "imports and checkpoints. Use this only for diagnostics."
        ),
    )
    parser.set_defaults(full=True)
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
    # Placement-only tests missed the exact class of failures we were seeing: an
    # offloaded module can load successfully, then legacy upstream code calls .to()
    # on an Accelerate `meta` placeholder during the first real inference. Exercise
    # the whole endpoint->flow->VAE->DDIM->decode path with one DDIM step instead.
    code = f"""
import gc
import sys
from pathlib import Path
from PIL import Image, ImageDraw
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
first = Image.new('RGBA', (96, 64), (0, 0, 0, 0))
second = Image.new('RGBA', (96, 64), (0, 0, 0, 0))
d0 = ImageDraw.Draw(first); d1 = ImageDraw.Draw(second)
d0.rectangle((12, 16, 44, 48), fill=(255, 80, 40, 255))
d1.rectangle((48, 16, 80, 48), fill=(255, 80, 40, 255))
frames = impl._generate_clip(
    torch, model, inference, first, second, (224, 128), ddim_steps=1
)
assert frames, 'MoG inference smoke produced no frames'
torch.cuda.synchronize()
print('MoG {variant} real inference smoke: ok', len(frames), 'frames',
      'offload=' + str(getattr(model, '_webp_offload_mode', 'gpu')),
      f'{{total_vram / 1024**3:.1f}}GB GPU')
del frames, model, inference
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
from PIL import Image, ImageDraw
root = Path({str(ROOT)!r})
sys.path.insert(0, str(root))
import tooncrafter_selective_worker_entry as entry
impl = entry.implementation
torch, device, model, total_vram = impl.load_model(
    Path({str(source)!r}),
    Path({str(config)!r}),
    Path({str(checkpoint)!r}),
)
first = Image.new('RGBA', (96, 64), (0, 0, 0, 0))
second = Image.new('RGBA', (96, 64), (0, 0, 0, 0))
d0 = ImageDraw.Draw(first); d1 = ImageDraw.Draw(second)
d0.ellipse((12, 14, 46, 50), fill=(60, 170, 255, 255))
d1.ellipse((48, 14, 82, 50), fill=(60, 170, 255, 255))
frames = impl._generate_clip(
    torch, device, model, first, second, (224, 128), ddim_steps=1
)
assert frames, 'ToonCrafter inference smoke produced no frames'
torch.cuda.synchronize()
print('ToonCrafter real inference smoke: ok', len(frames), 'frames',
      'offload=' + str(getattr(model, '_webp_offload_mode', 'gpu')),
      f'{{total_vram / 1024**3:.1f}}GB GPU')
del frames, model
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
        print("Quick mode skipped MoG/ToonCrafter real inference; do not treat this as release qualification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
