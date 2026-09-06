from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any


ACTIVE_STATES = {"queued", "running"}


def _now() -> float:
    return time.time()


def _valid_job_id(job_id: str) -> bool:
    return bool(job_id) and len(job_id) == 32 and all(ch in "0123456789abcdef" for ch in job_id.lower())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(min(0.01 * (attempt + 1), 0.10))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class GlobalJobStore:
    """Durable top-level WebP Animator jobs.

    A global job is the user's recoverable workspace. It can contain source frames,
    ordinary render state/result, and links to one or more OpenAI child jobs. Nothing
    is removed by age. A new workspace gets a new ID, while old IDs remain restorable.
    """

    def __init__(self, root: Path):
        self.root = root / ".webp-jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._mark_interrupted()

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def job_dir(self, job_id: str) -> Path:
        if not _valid_job_id(job_id):
            raise ValueError("Invalid global job ID.")
        return self.root / job_id.lower()

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def get(self, job_id: str) -> dict[str, Any] | None:
        try:
            path = self.manifest_path(job_id)
        except ValueError:
            return None
        with self.lock:
            if not path.is_file():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else None
            except Exception:
                return None

    def save(self, job: dict[str, Any]) -> dict[str, Any]:
        job = dict(job)
        job["updated"] = _now()
        with self.lock:
            _atomic_json(self.manifest_path(str(job["id"])), job)
        return job

    def ensure(self, job_id: str | None = None) -> dict[str, Any]:
        job_id = (job_id or self.new_id()).lower()
        if not _valid_job_id(job_id):
            raise ValueError("Invalid global job ID.")
        existing = self.get(job_id)
        if existing:
            return existing
        job = {
            "id": job_id,
            "kind": "global",
            "status": "draft",
            "progress": 0,
            "message": "Workspace saved",
            "error": None,
            "created": _now(),
            "updated": _now(),
            "settings": {},
            "sources": [],
            "output_path": None,
            "output_available": False,
            "render_count": 0,
            "openai_jobs": [],
        }
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        return self.save(job)

    def save_sources(
        self,
        job_id: str,
        uploads: list[tuple[str, bytes]],
        settings: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[Path]]:
        job = self.ensure(job_id)
        job_dir = self.job_dir(job_id)
        source_dir = job_dir / "source"
        staging = job_dir / f".source-{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        stored: list[dict[str, Any]] = []
        paths: list[Path] = []
        try:
            for index, (name, payload) in enumerate(uploads):
                suffix = Path(name).suffix.lower() or ".bin"
                path = staging / f"{index:06d}{suffix}"
                path.write_bytes(payload)
                stored.append({
                    "index": index,
                    "name": Path(name).name,
                    "path": str(Path("source") / path.name),
                    "size": len(payload),
                })
            with self.lock:
                old = job_dir / f".source-old-{uuid.uuid4().hex}"
                if source_dir.exists():
                    os.replace(source_dir, old)
                os.replace(staging, source_dir)
                shutil.rmtree(old, ignore_errors=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        job = self.get(job_id) or job
        job["sources"] = stored
        if settings is not None:
            job["settings"] = dict(settings)
        self.save(job)
        paths = [job_dir / str(item["path"]) for item in stored]
        return job, paths

    def begin_render(
        self,
        job_id: str,
        uploads: list[tuple[str, bytes]],
        settings: dict[str, Any],
    ) -> tuple[dict[str, Any], list[Path]]:
        job, paths = self.save_sources(job_id, uploads, settings)
        job = self.get(job_id) or job
        job.update({
            "status": "queued",
            "progress": 5,
            "message": "Upload received and persisted",
            "error": None,
            "render_count": int(job.get("render_count", 0)) + 1,
        })
        return self.save(job), paths

    def update(self, job_id: str, **changes: Any) -> dict[str, Any] | None:
        with self.lock:
            job = self.get(job_id)
            if not job:
                return None
            job.update(changes)
            return self.save(job)

    def attach_openai(self, job_id: str, openai_job_id: str) -> dict[str, Any]:
        job = self.ensure(job_id)
        linked = list(job.get("openai_jobs", []))
        if openai_job_id not in linked:
            linked.append(openai_job_id)
        job["openai_jobs"] = linked
        job["last_openai_job_id"] = openai_job_id
        return self.save(job)

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        output = self.job_dir(str(job["id"])) / "animation.webp"
        return {
            "id": job.get("id"),
            "kind": "global",
            "status": job.get("status", "draft"),
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "error": job.get("error"),
            "created": job.get("created"),
            "updated": job.get("updated"),
            "settings": job.get("settings", {}),
            "source_count": len(job.get("sources", [])),
            "source_names": [str(item.get("name", f"frame-{i + 1}.png")) for i, item in enumerate(job.get("sources", []))],
            "output_available": output.is_file(),
            "output_size": output.stat().st_size if output.is_file() else 0,
            "render_count": job.get("render_count", 0),
            "openai_jobs": list(job.get("openai_jobs", [])),
            "last_openai_job_id": job.get("last_openai_job_id"),
        }

    def source_info(self, job_id: str, index: int) -> tuple[Path, str] | None:
        job = self.get(job_id)
        if not job:
            return None
        sources = list(job.get("sources", []))
        if index < 0 or index >= len(sources):
            return None
        item = sources[index]
        path = self.job_dir(job_id) / str(item.get("path", ""))
        if not path.is_file():
            return None
        return path, str(item.get("name", f"frame-{index + 1}.png"))

    def output_path(self, job_id: str) -> Path | None:
        if not self.get(job_id):
            return None
        path = self.job_dir(job_id) / "animation.webp"
        return path if path.is_file() else None

    def active_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for manifest in self.root.glob("*/job.json"):
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
                if str(value.get("status", "")) in ACTIVE_STATES:
                    jobs.append(value)
            except Exception:
                continue
        return jobs

    def _mark_interrupted(self) -> None:
        for manifest in self.root.glob("*/job.json"):
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
                if str(value.get("status", "")) in ACTIVE_STATES:
                    value["status"] = "interrupted"
                    value["message"] = "Server restarted during processing. The persisted job can be rendered again without rebuilding the workspace."
                    value["updated"] = _now()
                    _atomic_json(manifest, value)
            except Exception:
                continue
