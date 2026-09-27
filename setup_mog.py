#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from setup_engine_common import ensure_repo, ensure_venv, run

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party" / "MoG-VFI"
VENV = ROOT / ".mog-venv"
REPOSITORY = "https://github.com/MCG-NJU/MoG-VFI.git"
MODEL_REPOSITORY = "MCG-NJU/MoG"
FLOW_FOLDER = "https://drive.google.com/drive/folders/16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o"


def parse_args():
    parser = argparse.ArgumentParser(description="Install Motion-Aware Generative Frame Interpolation")
    parser.add_argument("--variant", choices=("ani", "real", "both"), default="ani")
    return parser.parse_args()


def choose_python() -> str:
    explicit = os.environ.get("MOG_BOOTSTRAP_PYTHON")
    candidates = [explicit] if explicit else []
    candidates += ["python3.10", "python3.9", "python3.8", sys.executable]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if resolved and Path(resolved).is_file():
            return str(resolved)
    raise RuntimeError("Could not find Python 3.8-3.10 for the MoG environment.")


def install_requirements(python: Path) -> None:
    run([
        python,
        "-m",
        "pip",
        "install",
        "torch==2.1.0+cu121",
        "torchvision==0.16.0+cu121",
        "--index-url",
        os.environ.get("MOG_TORCH_INDEX", "https://download.pytorch.org/whl/cu121"),
    ])

    # The published requirements include training-only DeepSpeed/LoRA packages.
    # They are unnecessary for the released inference path and are troublesome on
    # Windows, so keep the upstream inference stack while omitting those three lines.
    source_requirements = (SOURCE / "requirements.txt").read_text(encoding="utf-8")
    kept = []
    for raw in source_requirements.splitlines():
        line = raw.strip()
        lower = line.lower()
        if not line:
            continue
        if lower.startswith("deepspeed"):
            continue
        if lower.startswith("accelerate[deepspeed]"):
            continue
        if "microsoft/lora" in lower:
            continue
        kept.append(line)

    with tempfile.TemporaryDirectory(prefix="mog_requirements_") as temp:
        requirements = Path(temp) / "requirements-inference.txt"
        requirements.write_text("\n".join(kept) + "\n", encoding="utf-8")
        run([python, "-m", "pip", "install", "-r", requirements])

    run([python, "-m", "pip", "install", "gdown", "huggingface_hub>=0.25,<1"])


def download_model(python: Path, variant: str) -> None:
    checkpoint_dir = SOURCE / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / f"{variant}.ckpt"
    if destination.is_file() and destination.stat().st_size > 1024 * 1024:
        print(f"Already present: {destination}")
        return
    code = (
        "from huggingface_hub import hf_hub_download; "
        f"print(hf_hub_download(repo_id={MODEL_REPOSITORY!r}, filename={variant + '.ckpt'!r}, "
        f"local_dir={str(checkpoint_dir)!r}))"
    )
    run([python, "-c", code])


def download_flow_checkpoint(python: Path) -> None:
    checkpoint_dir = SOURCE / "emavfi" / "ckpt"
    upstream_destination = checkpoint_dir / "ours_t.pkl"
    compatibility_destination = checkpoint_dir / "ours_t.ckpt"

    # Upstream's README calls this file ours_t.ckpt, but the Google Drive folder
    # actually contains ours_t.pkl and emavfi/Trainer.py loads ours_t.pkl. Keep the
    # real upstream filename and a compatibility copy for our existing worker path.
    if upstream_destination.is_file() and upstream_destination.stat().st_size > 1024 * 1024:
        if (
            not compatibility_destination.is_file()
            or compatibility_destination.stat().st_size != upstream_destination.stat().st_size
        ):
            shutil.copy2(upstream_destination, compatibility_destination)
        print(f"Already present: {upstream_destination}")
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mog_flow_") as temp:
        temp_dir = Path(temp)
        run([python, "-m", "gdown", "--folder", FLOW_FOLDER, "-O", temp_dir])
        candidates = list(temp_dir.rglob("ours_t.pkl"))
        if not candidates:
            # Retain compatibility with a future upstream folder correction.
            candidates = list(temp_dir.rglob("ours_t.ckpt"))
        if not candidates:
            raise RuntimeError(
                "MoG flow checkpoint download completed but neither ours_t.pkl nor "
                "ours_t.ckpt was found. Set MOG_FLOW_CHECKPOINT to an existing copy "
                "if Google Drive blocks the folder download."
            )
        shutil.copy2(candidates[0], upstream_destination)
        shutil.copy2(candidates[0], compatibility_destination)


def main() -> None:
    args = parse_args()
    print("Installing Motion-Aware Generative Frame Interpolation")
    ensure_repo(REPOSITORY, SOURCE)
    python = ensure_venv(VENV, choose_python())
    install_requirements(python)

    variants = ("ani", "real") if args.variant == "both" else (args.variant,)
    for variant in variants:
        download_model(python, variant)
    download_flow_checkpoint(python)

    print("\nMoG setup complete.")
    print(f"Python: {python}")
    print(f"Source: {SOURCE}")
    print(f"Installed variant(s): {', '.join(variants)}")
    print("The released MoG checkpoints are very large. 4 GB GPUs may still fail even at the lowest fallback resolution.")
    print("Restart app_all.py if it is already running.")


if __name__ == "__main__":
    main()
