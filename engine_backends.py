#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent


def _venv_python(dirname: str) -> Path:
    base = ROOT / dirname
    return base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def engine_status() -> dict[str, dict[str, object]]:
    specs = {
        "rife": {
            "python": _venv_python(".rife-venv"),
            "source": ROOT / "third_party" / "Practical-RIFE",
            "checkpoint": ROOT / "third_party" / "Practical-RIFE" / "train_log" / "flownet.pkl",
            "setup": "python setup_rife.py",
        },
        "amt": {
            "python": _venv_python(".amt-venv"),
            "source": ROOT / "third_party" / "AMT",
            "checkpoint": ROOT / "third_party" / "AMT" / "pretrained" / "amt-s.pth",
            "setup": "python setup_amt.py",
        },
        "eden": {
            "python": _venv_python(".eden-venv"),
            "source": ROOT / "third_party" / "EDEN",
            "checkpoint": ROOT / "third_party" / "EDEN" / "checkpoints" / "eden.pt",
            "setup": "python setup_eden.py",
        },
        "speed": {
            "python": _venv_python(".speed-venv"),
            "source": ROOT / "third_party" / "SPEED",
            "checkpoint": ROOT / "third_party" / "SPEED" / "ckpts" / "speed.pt",
            "setup": "python setup_speed.py",
        },
    }
    result: dict[str, dict[str, object]] = {}
    for name, spec in specs.items():
        python = Path(spec["python"])
        source = Path(spec["source"])
        checkpoint = Path(spec["checkpoint"])
        result[name] = {
            "ready": python.is_file() and source.is_dir() and checkpoint.is_file(),
            "python": str(python),
            "source": str(source),
            "checkpoint": str(checkpoint),
            "setup": spec["setup"],
        }
    return result


def require_engine(name: str) -> dict[str, object]:
    status = engine_status().get(name)
    if not status:
        raise RuntimeError(f"Unknown engine: {name}")
    if not status["ready"]:
        raise RuntimeError(f"{name.upper()} is not installed. Run `{status['setup']}` and restart the server.")
    return status


def run_worker(
    *,
    name: str,
    worker: str,
    input_dir: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    status = require_engine(name)
    command = [
        str(status["python"]),
        str(ROOT / worker),
        f"--{name}-dir", str(status["source"]),
        "--checkpoint", str(status["checkpoint"]),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    ]
    if extra_args:
        command.extend(extra_args)

    output_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip()
        if line.startswith("PROGRESS "):
            try:
                _, current, total = line.split()
                if progress:
                    progress(int(current), max(1, int(total)))
            except (ValueError, IndexError):
                pass
        elif line:
            tail.append(line)
            tail = tail[-30:]

    code = process.wait()
    if code != 0:
        details = "\n".join(tail[-12:]) or f"{name.upper()} worker produced no diagnostic output."
        raise RuntimeError(f"{name.upper()} failed:\n{details}")
