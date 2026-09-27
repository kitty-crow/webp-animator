from __future__ import annotations

from pathlib import Path

import generative_vfi
from engine_paths import tooncrafter_paths

ROOT = Path(__file__).resolve().parent
_installed = False


def install_backend() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    generative_vfi.MARKERS["tooncrafter"] = "__vfi_tooncrafter__"
    generative_vfi.LABELS["tooncrafter"] = "ToonCrafter"

    original = generative_vfi._engine_command

    def engine_command(engine: str, manifest: Path, result: Path):
        if engine != "tooncrafter":
            return original(engine, manifest, result)

        ready, python, source, config, checkpoint = tooncrafter_paths()
        if not ready:
            raise RuntimeError(
                "ToonCrafter is selected but is not installed. "
                "Run `python setup_tooncrafter.py`."
            )
        return [
            python,
            ROOT / "tooncrafter_selective_worker_entry.py",
            "--tooncrafter-dir", source,
            "--config", config,
            "--checkpoint", checkpoint,
            "--manifest", manifest,
            "--result", result,
        ], source

    generative_vfi._engine_command = engine_command
