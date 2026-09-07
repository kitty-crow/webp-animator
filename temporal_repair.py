from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parent
REPAIR_TOKEN = "repair:propainter"
_tls = threading.local()


def repair_requested(settings: dict | None) -> bool:
    if not isinstance(settings, dict):
        return False
    tokens = {
        part.strip().lower()
        for part in str(settings.get("target_gaps", "")).replace(";", ",").split(",")
        if part.strip()
    }
    return REPAIR_TOKEN in tokens


def _venv_python(venv: Path) -> Path:
    candidates = (
        venv / "Scripts" / "python.exe",
        venv / "Scripts" / "python",
        venv / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if os.name == "nt" else candidates[-1]


def propainter_paths():
    python = Path(os.environ.get("PROPAINTER_PYTHON", str(_venv_python(ROOT / ".propainter-venv")))).expanduser()
    source = Path(os.environ.get("PROPAINTER_DIR", str(ROOT / "third_party" / "ProPainter"))).expanduser()
    weights = source / "weights"
    required = (
        weights / "ProPainter.pth",
        weights / "recurrent_flow_completion.pth",
        weights / "raft-things.pth",
    )
    ready = python.is_file() and source.is_dir() and all(path.is_file() for path in required)
    return ready, python, source


def _save_input_frames(images: Sequence[Image.Image], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, image in enumerate(images):
        path = directory / f"{index:06d}.png"
        image.convert("RGBA").save(path)
        paths.append(path)
    return paths


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    # Older research-model torch builds can reject this newer allocator option
    # before CUDA even initialises. The repair worker is short-lived, so remove
    # only that unsupported option at the process boundary.
    raw = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if raw:
        pieces = [item.strip() for item in raw.split(",") if item.strip()]
        pieces = [item for item in pieces if not item.lower().startswith("expandable_segments:")]
        if pieces:
            env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(pieces)
        else:
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def repair_images(
    images: Sequence[Image.Image],
    target_indexes: Sequence[int],
    work_dir: Path,
    *,
    progress=None,
):
    targets = sorted({int(index) for index in target_indexes if 0 <= int(index) < len(images)})
    if not targets:
        return [image.convert("RGBA").copy() for image in images], {
            "audited": 0,
            "repaired": 0,
            "mask_pixels": 0,
            "engine": "propainter",
        }

    ready, python, source = propainter_paths()
    if not ready:
        raise RuntimeError(
            "Temporal frame repair is enabled but ProPainter is not installed. "
            "Run `python setup_propainter.py` on JASPER."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    result_path = work_dir / "result.json"
    manifest_path = work_dir / "manifest.json"
    frame_paths = _save_input_frames(images, input_dir)
    manifest_path.write_text(
        json.dumps(
            {
                "frames": [str(path) for path in frame_paths],
                "targets": targets,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    command = [
        str(python),
        str(ROOT / "propainter_repair_worker.py"),
        "--propainter-dir",
        str(source),
        "--manifest",
        str(manifest_path),
        "--result",
        str(result_path),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(source),
        env=_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if line.startswith("PROGRESS "):
            try:
                _, current, total = line.split()
                if progress:
                    progress(int(current), max(1, int(total)))
            except Exception:
                pass
        elif line:
            tail.append(line)
            tail = tail[-40:]
    code = process.wait()
    if code != 0:
        details = "\n".join(tail[-16:]) or "No ProPainter worker output."
        raise RuntimeError(f"ProPainter temporal repair failed:\n{details}")
    if not result_path.is_file():
        raise RuntimeError("ProPainter temporal repair did not return a result manifest.")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    repaired = [image.convert("RGBA").copy() for image in images]
    for item in result.get("frames", []):
        index = int(item.get("index", -1))
        path = Path(str(item.get("path", "")))
        if index < 0 or index >= len(repaired) or not path.is_file():
            continue
        with Image.open(path) as image:
            repaired[index] = image.convert("RGBA").copy()

    stats = {
        "audited": len(targets),
        "repaired": int(result.get("repaired", 0)),
        "mask_pixels": int(result.get("mask_pixels", 0)),
        "engine": "propainter",
        "cuda": bool(result.get("cuda", False)),
        "device": result.get("device"),
        "processing_size": result.get("processing_size"),
    }
    return repaired, stats


def install(temporal_v2_module):
    if getattr(temporal_v2_module, "_propainter_temporal_repair_installed", False):
        return
    temporal_v2_module._propainter_temporal_repair_installed = True

    original_process_job = temporal_v2_module.process_job
    original_insert_gap_frames = temporal_v2_module._insert_gap_frames
    original_save_webp_fast = temporal_v2_module.save_webp_fast

    def insert_gap_frames(records, *args, **kwargs):
        result = original_insert_gap_frames(records, *args, **kwargs)
        # Mark the actual PIL objects. The encoder receives these same image
        # objects later, allowing the repair stage to target generated frames
        # without changing the established FrameRecord API.
        for record in records:
            if getattr(record, "generated", False):
                record.image.info["_webp_temporal_generated"] = "1"
        return result

    def save_webp_with_repair(images, output_path, *args, **kwargs):
        settings = getattr(_tls, "settings", None)
        if not repair_requested(settings):
            return original_save_webp_fast(images, output_path, *args, **kwargs)

        target_indexes = [
            index
            for index, image in enumerate(images)
            if str(getattr(image, "info", {}).get("_webp_temporal_generated", "")) == "1"
        ]
        if not target_indexes:
            _tls.repair_stats = {
                "audited": 0,
                "repaired": 0,
                "mask_pixels": 0,
                "engine": "propainter",
            }
            return original_save_webp_fast(images, output_path, *args, **kwargs)

        legacy = getattr(_tls, "legacy", None)
        job_id = getattr(_tls, "job_id", "")

        def report(current, total):
            if legacy is None or not job_id:
                return
            fraction = current / max(1, total)
            legacy.set_job(
                job_id,
                progress=min(97, 88 + round(fraction * 9)),
                message=f"ProPainter temporal repair · step {current}/{total}",
            )

        repaired, stats = repair_images(
            images,
            target_indexes,
            Path(output_path).parent / "advanced" / "temporal-repair",
            progress=report,
        )
        _tls.repair_stats = stats
        return original_save_webp_fast(repaired, output_path, *args, **kwargs)

    def process_job(legacy, job_id, paths, settings):
        _tls.settings = dict(settings or {})
        _tls.legacy = legacy
        _tls.job_id = job_id
        _tls.repair_stats = None
        try:
            result = original_process_job(legacy, job_id, paths, settings)
            stats = getattr(_tls, "repair_stats", None)
            if repair_requested(settings) and stats:
                job = legacy.get_job(job_id) or {}
                if str(job.get("status", "")) == "done":
                    message = str(job.get("message", "Ready"))
                    message += (
                        f"; ProPainter audited {stats['audited']} generated frame"
                        f"{'s' if stats['audited'] != 1 else ''} and repaired {stats['repaired']}"
                    )
                    legacy.set_job(job_id, message=message, temporal_repair=stats)
            return result
        finally:
            for name in ("settings", "legacy", "job_id", "repair_stats"):
                try:
                    delattr(_tls, name)
                except AttributeError:
                    pass

    temporal_v2_module._insert_gap_frames = insert_gap_frames
    temporal_v2_module.save_webp_fast = save_webp_with_repair
    temporal_v2_module.process_job = process_job
