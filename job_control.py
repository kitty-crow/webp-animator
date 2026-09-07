from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parent
LIVE_UI = ROOT / "live_run_ui.js"

_lock = threading.RLock()
_events: dict[str, threading.Event] = {}
_processes: dict[str, set[subprocess.Popen]] = {}
_tls = threading.local()
_installed = False
_legacy = None
_raw_set_job = None


class JobCancelled(RuntimeError):
    pass


def _event(job_id: str) -> threading.Event:
    with _lock:
        event = _events.get(job_id)
        if event is None:
            event = threading.Event()
            _events[job_id] = event
        return event


def begin(job_id: str) -> None:
    job_id = str(job_id or "")
    event = _event(job_id)
    event.clear()
    with _lock:
        stale = list(_processes.pop(job_id, set()))
    for process in stale:
        _terminate(process)
    _tls.job_id = job_id


def end(job_id: str) -> None:
    if getattr(_tls, "job_id", None) == job_id:
        try:
            delattr(_tls, "job_id")
        except AttributeError:
            pass
    with _lock:
        _processes.pop(job_id, None)


def current_job_id() -> str:
    return str(getattr(_tls, "job_id", "") or "")


def is_cancelled(job_id: str | None = None) -> bool:
    job_id = str(job_id or current_job_id())
    return bool(job_id) and _event(job_id).is_set()


def raise_if_cancelled(job_id: str | None = None) -> None:
    job_id = str(job_id or current_job_id())
    if job_id and is_cancelled(job_id):
        raise JobCancelled("Stopped by user.")


def _terminate(process: subprocess.Popen) -> None:
    try:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.8)
            return
        except subprocess.TimeoutExpired:
            pass
        process.kill()
    except Exception:
        pass


def register_process(process: subprocess.Popen, job_id: str | None = None) -> str:
    job_id = str(job_id or current_job_id())
    if not job_id:
        return ""
    with _lock:
        _processes.setdefault(job_id, set()).add(process)
        cancelled = _event(job_id).is_set()
    if cancelled:
        _terminate(process)
    return job_id


def unregister_process(process: subprocess.Popen, job_id: str | None = None) -> None:
    job_id = str(job_id or current_job_id())
    if not job_id:
        return
    with _lock:
        group = _processes.get(job_id)
        if group is not None:
            group.discard(process)
            if not group:
                _processes.pop(job_id, None)


def cancel(job_id: str) -> int:
    job_id = str(job_id or "")
    if not job_id:
        return 0
    event = _event(job_id)
    event.set()
    with _lock:
        processes = list(_processes.get(job_id, set()))
    for process in processes:
        _terminate(process)
    return len(processes)


def run_process(command, *, cwd: Path, progress_callback=None, label: str):
    """Drop-in replacement for advanced_pipeline._run_process with cancellation."""
    raise_if_cancelled()
    process = subprocess.Popen(
        [str(item) for item in command],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    job_id = register_process(process)
    tail: list[str] = []
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            raise_if_cancelled(job_id)
            line = raw.strip()
            if line.startswith("PROGRESS "):
                try:
                    _, current, total = line.split()
                    if progress_callback:
                        progress_callback(int(current), max(1, int(total)))
                except JobCancelled:
                    raise
                except Exception:
                    pass
            elif line:
                tail.append(line)
                tail = tail[-30:]
        code = process.wait()
        raise_if_cancelled(job_id)
        if code != 0:
            details = "\n".join(tail[-12:]) or "No worker output."
            raise RuntimeError(f"{label} failed:\n{details}")
    finally:
        unregister_process(process, job_id)
        if is_cancelled(job_id):
            _terminate(process)


def _candidate_live_frame(relative: Path) -> bool:
    parts = [part.lower() for part in relative.parts]
    if not parts:
        return False
    first = parts[0]
    if "generator" in first or "interpolator" in first:
        return True
    if first == "temporal-repair" and len(parts) >= 2 and parts[1] == "output":
        return True
    return False


def _live_root(job_id: str) -> tuple[Path | None, dict | None]:
    if _legacy is None:
        return None, None
    job = _legacy.get_job(job_id)
    if not isinstance(job, dict):
        return None, None
    job_dir = Path(str(job.get("job_dir", "")))
    if not job_dir:
        return None, job
    return job_dir / "advanced", job


def _live_frames(job_id: str) -> tuple[list[dict], dict | None]:
    root, job = _live_root(job_id)
    if root is None or not root.is_dir():
        return [], job
    items: list[dict] = []
    for path in root.rglob("*.png"):
        try:
            relative = path.relative_to(root)
            if not _candidate_live_frame(relative):
                continue
            stat = path.stat()
            if stat.st_size <= 0:
                continue
            rel = relative.as_posix()
            first = relative.parts[0] if relative.parts else "generated"
            if first == "temporal-repair":
                stage = "ProPainter repair"
            else:
                stage = first.replace("-v2", "").replace("-", " ").strip()
            url = (
                f"/live-frame?id={quote(job_id)}&rel={quote(rel)}"
                f"&v={stat.st_mtime_ns}"
            )
            items.append(
                {
                    "key": rel,
                    "name": path.name,
                    "stage": stage,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "url": url,
                }
            )
        except OSError:
            continue
    items.sort(key=lambda item: (int(item["mtime_ns"]), str(item["key"])))
    return items, job


def _serve_live_frame(handler, job_id: str, relative_text: str) -> None:
    root, _ = _live_root(job_id)
    if root is None:
        handler.send_text(404, "Unknown job.")
        return
    try:
        root_resolved = root.resolve()
        target = (root / relative_text).resolve()
        relative = target.relative_to(root_resolved)
    except Exception:
        handler.send_text(400, "Invalid live-frame path.")
        return
    if not _candidate_live_frame(relative) or target.suffix.lower() != ".png" or not target.is_file():
        handler.send_text(404, "Live frame not found.")
        return
    try:
        payload = target.read_bytes()
    except OSError:
        handler.send_text(409, "Frame is still being written; retry shortly.")
        return
    handler.send_bytes(
        200,
        payload,
        "image/png",
        {
            "Cache-Control": "no-store, max-age=0",
            "Content-Disposition": f'inline; filename="{target.name}"',
        },
    )


def install(legacy, advanced_pipeline_module, temporal_v2_module) -> None:
    global _installed, _legacy, _raw_set_job
    if _installed:
        return
    _installed = True
    _legacy = legacy
    _raw_set_job = legacy.set_job

    # Once a job has been stopped, late progress/error callbacks from the render
    # thread must not resurrect it as running/error/done.
    def guarded_set_job(job_id, **changes):
        current = legacy.get_job(job_id)
        if isinstance(current, dict) and str(current.get("status", "")) == "cancelled":
            return current
        return _raw_set_job(job_id, **changes)

    legacy.set_job = guarded_set_job

    # All RIFE/AMT/EDEN/SPEED launches in temporal_v2 pass through this helper.
    advanced_pipeline_module._run_process = run_process

    original_process_job = temporal_v2_module.process_job

    def controlled_process_job(legacy_module, job_id, paths, settings):
        begin(job_id)
        try:
            return original_process_job(legacy_module, job_id, paths, settings)
        finally:
            if is_cancelled(job_id) and _raw_set_job is not None:
                _raw_set_job(
                    job_id,
                    status="cancelled",
                    message="Stopped by user. Generated frames produced before the stop were kept for inspection.",
                    error=None,
                )
            end(job_id)

    temporal_v2_module.process_job = controlled_process_job

    original_get = legacy.Handler.do_GET
    original_post = legacy.Handler.do_POST

    def do_get(handler):
        parsed = urlparse(handler.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/live-run-ui.js":
            if not LIVE_UI.is_file():
                handler.send_text(404, "Live run UI is missing.")
                return
            handler.send_bytes(200, LIVE_UI.read_bytes(), "text/javascript; charset=utf-8", {"Cache-Control": "no-store"})
            return
        if parsed.path == "/live-frames":
            job_id = str(query.get("id", [""])[0]).strip().lower()
            frames, job = _live_frames(job_id)
            if job is None:
                handler.send_json(404, {"error": "Unknown job."})
                return
            handler.send_json(
                200,
                {
                    "job_id": job_id,
                    "status": job.get("status", "draft"),
                    "render_count": int(job.get("render_count", 0)),
                    "frames": frames,
                },
            )
            return
        if parsed.path == "/live-frame":
            job_id = str(query.get("id", [""])[0]).strip().lower()
            relative = str(query.get("rel", [""])[0])
            _serve_live_frame(handler, job_id, relative)
            return
        return original_get(handler)

    def do_post(handler):
        parsed = urlparse(handler.path)
        if parsed.path == "/job/stop":
            query = parse_qs(parsed.query)
            job_id = str(query.get("id", [""])[0]).strip().lower()
            job = legacy.get_job(job_id)
            if not isinstance(job, dict):
                handler.send_json(404, {"error": "Unknown job."})
                return
            status = str(job.get("status", "draft"))
            if status == "cancelled":
                handler.send_json(200, {"status": "cancelled", "message": job.get("message", "Stopped by user.")})
                return
            if status not in {"queued", "running"}:
                handler.send_json(409, {"error": f"Job is not running ({status})."})
                return
            terminated = cancel(job_id)
            if _raw_set_job is not None:
                _raw_set_job(
                    job_id,
                    status="cancelled",
                    message="Stopped by user. Generated frames produced before the stop were kept for inspection.",
                    error=None,
                )
            handler.send_json(
                200,
                {
                    "status": "cancelled",
                    "message": "Stopped by user.",
                    "terminated_workers": terminated,
                },
            )
            return
        return original_post(handler)

    legacy.Handler.do_GET = do_get
    legacy.Handler.do_POST = do_post
