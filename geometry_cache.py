from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Sequence

from PIL import Image


CACHE_VERSION = "geometry-cache-v1"
_LOCK = threading.RLock()
_GEOMETRY_KEYS = (
    "geometry_mode",
    "axis",
    "max_shift",
    "sigma",
    "alpha_threshold",
)


def geometry_settings(settings: dict | None) -> dict:
    value = settings if isinstance(settings, dict) else {}
    return {key: value.get(key) for key in _GEOMETRY_KEYS}


def cache_key(paths: Sequence[Path], settings: dict | None) -> str:
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(geometry_settings(settings), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    digest.update(b"\0")
    for ordinal, path in enumerate(paths):
        digest.update(str(ordinal).encode("ascii"))
        digest.update(b":")
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _entry(cache_root: Path, key: str) -> Path:
    return Path(cache_root) / key


def load(cache_root: Path, key: str) -> list[Image.Image] | None:
    root = _entry(cache_root, key)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if value.get("version") != CACHE_VERSION or value.get("key") != key:
            return None
        count = int(value.get("frame_count", -1))
        if count < 0:
            return None
        frames: list[Image.Image] = []
        for index in range(count):
            path = root / f"{index:06d}.png"
            if not path.is_file():
                return None
            with Image.open(path) as image:
                frames.append(image.convert("RGBA").copy())
        return frames
    except Exception:
        return None


def save(cache_root: Path, key: str, frames: Sequence[Image.Image], settings: dict | None) -> Path:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = _entry(cache_root, key)
    with _LOCK:
        if (target / "manifest.json").is_file():
            return target
        staging = cache_root / f".{key}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for index, image in enumerate(frames):
                image.convert("RGBA").save(staging / f"{index:06d}.png", format="PNG")
            (staging / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": CACHE_VERSION,
                        "key": key,
                        "frame_count": len(frames),
                        "geometry": geometry_settings(settings),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            try:
                os.replace(staging, target)
            except (FileExistsError, PermissionError, OSError):
                if not (target / "manifest.json").is_file():
                    raise
            return target
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def normalise_with_cache(
    legacy,
    images: list[Image.Image],
    source_paths: Sequence[Path],
    settings: dict,
    cache_root: Path | None,
    progress=None,
    *,
    cache_label: str = "gap analysis",
) -> tuple[list[Image.Image], dict]:
    """Normalise geometry once and optionally persist/reuse its exact output.

    Cache identity includes source bytes in order plus every setting consumed by
    `_normalise_geometry`. A changed frame, ordering, or geometry option therefore
    cannot accidentally reuse an old alignment.
    """
    import advanced_pipeline as base

    if cache_root is None:
        return base._normalise_geometry(legacy, images, settings, progress), {"hit": False, "key": None}

    key = cache_key(source_paths, settings)
    cached = load(cache_root, key)
    if cached is not None and len(cached) == len(images):
        if progress:
            progress(40, f"Reusing frame matching from {cache_label}")
        return cached, {"hit": True, "key": key}

    frames = base._normalise_geometry(legacy, images, settings, progress)
    save(cache_root, key, frames, settings)
    return frames, {"hit": False, "key": key}
