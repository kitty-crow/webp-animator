#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import threading
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import app as legacy
import advanced_pipeline
import gpu_match
import temporal_timing
import temporal_v2
from engine_paths import engine_status
from frame1_optimizer import find_best_scale_and_translation as fast_find_best_scale_and_translation
from global_jobs import GlobalJobStore

ROOT = Path(__file__).resolve().parent
GLOBAL = GlobalJobStore(ROOT)
WORKSPACE_FRAGMENT = (ROOT / "workspace_ui" / "fragment.html").read_text(encoding="utf-8")
WORKSPACE_SCRIPT = (ROOT / "workspace_ui" / "workspace.js").read_bytes()
ADVANCED_SCRIPT = (ROOT / "advanced_ui.js").read_bytes()

MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


def enhanced_index() -> bytes:
    html = legacy.INDEX_HTML.decode("utf-8")
    marker = '    <form id="form" enctype="multipart/form-data">'
    if marker not in html:
        raise RuntimeError("Could not find WebP Animator form for global-job UI injection.")
    html = html.replace(marker, WORKSPACE_FRAGMENT + "\n\n" + marker, 1)
    html = html.replace(
        "</body>",
        '  <script src="/workspace-ui.js"></script>\n  <script src="/advanced-ui.js"></script>\n</body>',
        1,
    )
    return html.encode("utf-8")


INDEX_HTML = enhanced_index()

_original_get_job = legacy.get_job
_original_set_job = legacy.set_job

# Keep the fast bounded CPU frame-1 optimiser as the fallback implementation.
legacy.find_best_scale_and_translation = fast_find_best_scale_and_translation


def _persistent_get_job(job_id: str):
    job = GLOBAL.get(job_id)
    if job:
        value = dict(job)
        value["job_dir"] = str(GLOBAL.job_dir(job_id))
        output = GLOBAL.output_path(job_id)
        if output:
            value["output_path"] = str(output)
        return value
    return _original_get_job(job_id)


def _persistent_set_job(job_id: str, **changes):
    if GLOBAL.get(job_id):
        GLOBAL.update(job_id, **changes)
        return
    _original_set_job(job_id, **changes)


legacy.get_job = _persistent_get_job
legacy.set_job = _persistent_set_job
legacy.clean_old_jobs = lambda: None

# Patch the existing geometry entry points only when CUDA PyTorch is available.
# Every GPU wrapper falls straight back to the established CPU implementation on
# any CUDA/OOM failure, so acceleration never becomes a hard runtime dependency.
GPU_GEOMETRY = False
try:
    GPU_GEOMETRY = gpu_match.install(legacy)
except Exception:
    GPU_GEOMETRY = False

# Keep the old module's timing patch for backwards-compatible direct imports/tests.
temporal_timing.install(advanced_pipeline)


def _advanced_process_job(job_id: str, paths: list[Path], settings: dict):
    return temporal_v2.process_job(legacy, job_id, paths, settings)


legacy.process_job = _advanced_process_job


def _settings_from_fields(fields: dict[str, str]) -> dict:
    axis = fields.get("axis", "xy")
    if axis not in {"x", "y", "xy", "none"}:
        axis = "xy"

    rife_multiplier = legacy.clamp_int(fields.get("rife_multiplier"), 1, 1, 8)
    if rife_multiplier not in {1, 2, 4, 8}:
        rife_multiplier = 1

    geometry_mode = fields.get("geometry_mode", "sequential")
    if geometry_mode not in {"none", "sequential", "fit_previous", "fix_first"}:
        geometry_mode = "sequential"
    if fields.get("shrink_larger") == "on" and geometry_mode == "sequential":
        geometry_mode = "fit_previous"

    interpolator = str(fields.get("interpolator", "")).strip().lower()
    if interpolator not in {"none", "rife", "amt"}:
        # Backward compatibility for persisted jobs created before the selector existed.
        interpolator = "rife" if rife_multiplier > 1 else "none"

    frame_generator = str(fields.get("frame_generator", "none")).strip().lower()
    if frame_generator not in {"none", "eden", "speed"}:
        frame_generator = "none"

    loop_analysis = str(fields.get("loop_analysis", "off")).strip().lower()
    if loop_analysis not in {"off", "alongside", "only"}:
        loop_analysis = "off"

    settings = {
        "axis": axis,
        "max_shift": legacy.clamp_int(fields.get("max_shift"), 64, 0, 2000),
        "duration": legacy.clamp_int(fields.get("duration"), 100, 1, 60000),
        "sigma": legacy.clamp_float(fields.get("sigma"), 24.0, 0.1, 255.0),
        "alpha_threshold": legacy.clamp_int(fields.get("alpha_threshold"), 8, 0, 255),
        "quality": legacy.clamp_int(fields.get("quality"), 90, 0, 100),
        "lossy": fields.get("lossy") == "on",
        "geometry_mode": geometry_mode,
        "rife_multiplier": rife_multiplier,
        "interpolator": interpolator,
        "frame_generator": frame_generator,
        "smart_reduction": fields.get("smart_reduction") == "on",
        "reduction_threshold": legacy.clamp_float(fields.get("reduction_threshold"), 2.0, 0.0, 100.0),
        "smart_missing": fields.get("smart_missing") == "on",
        "missing_threshold": legacy.clamp_float(fields.get("missing_threshold"), 12.0, 0.0, 100.0),
        "target_gaps": str(fields.get("target_gaps", "")).strip(),
        "loop_analysis": loop_analysis,
    }
    # Saved jobs from before this control existed retain the established
    # interpolation-density behaviour instead of silently becoming one-frame fills.
    if "frames_to_fill" in fields:
        try:
            settings["frames_to_fill"] = max(0, int(str(fields.get("frames_to_fill", "1")).strip()))
        except ValueError:
            settings["frames_to_fill"] = 1
    return settings


def _run_global_render(job_id: str, paths: list[Path], settings: dict) -> None:
    job_dir = GLOBAL.job_dir(job_id)
    shutil.rmtree(job_dir / "aligned", ignore_errors=True)
    shutil.rmtree(job_dir / "interpolated", ignore_errors=True)
    shutil.rmtree(job_dir / "advanced", ignore_errors=True)
    try:
        (job_dir / "animation.webp").unlink(missing_ok=True)
    except OSError:
        pass
    GLOBAL.update(job_id, status="running", progress=5, message="Persisted job started", error=None)
    legacy.process_job(job_id, paths, settings)


def _start_global_render(job_id: str, paths: list[Path], settings: dict) -> None:
    threading.Thread(
        target=_run_global_render,
        args=(job_id, paths, settings),
        daemon=True,
        name=f"webp-global-{job_id[:8]}",
    ).start()


def _analysis_update(job_id: str, analysis_id: str, **changes):
    job = GLOBAL.get(job_id)
    if not job:
        return None
    state = dict(job.get("analysis") or {})
    if state.get("id") and state.get("id") != analysis_id:
        return None
    state.update(changes)
    state["id"] = analysis_id
    GLOBAL.update(job_id, analysis=state)
    return state


def _stage_analysis_sources(
    job_id: str,
    analysis_id: str,
    uploads: list[tuple[str, bytes]],
) -> tuple[list[Path], list[dict]]:
    """Persist an analysis-only source snapshot without touching render sources."""
    GLOBAL.ensure(job_id)
    root = GLOBAL.job_dir(job_id) / "analysis" / analysis_id / "source"
    root.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    sources: list[dict] = []
    try:
        for index, (name, payload) in enumerate(uploads):
            suffix = Path(name).suffix.lower() or ".bin"
            path = root / f"{index:06d}{suffix}"
            path.write_bytes(payload)
            paths.append(path)
            sources.append(
                {
                    "index": index,
                    "name": Path(name).name,
                    "path": str(path.relative_to(GLOBAL.job_dir(job_id))),
                    "size": len(payload),
                }
            )
        return paths, sources
    except Exception:
        shutil.rmtree(root.parent, ignore_errors=True)
        raise


def _analysis_paths(job_id: str, analysis: dict) -> list[Path]:
    job_dir = GLOBAL.job_dir(job_id)
    sources = list(analysis.get("sources", []))
    return [job_dir / str(item.get("path", "")) for item in sources]


def _cleanup_analysis_snapshot(job_id: str, analysis_id: str) -> None:
    shutil.rmtree(GLOBAL.job_dir(job_id) / "analysis" / analysis_id, ignore_errors=True)


def _run_analysis(job_id: str, analysis_id: str, paths: list[Path], settings: dict) -> None:
    _analysis_update(
        job_id,
        analysis_id,
        status="running",
        progress=1,
        message="Server-side analysis started",
        error=None,
    )

    def report(value, message):
        _analysis_update(
            job_id,
            analysis_id,
            status="running",
            progress=max(0, min(100, int(value))),
            message=str(message),
            error=None,
        )

    try:
        result = temporal_v2.analyse_paths(legacy, paths, settings, progress=report)
        _analysis_update(
            job_id,
            analysis_id,
            status="done",
            progress=100,
            message="Analysis complete",
            error=None,
            result=result,
            sources=[],
        )
    except Exception as exc:
        _analysis_update(
            job_id,
            analysis_id,
            status="error",
            message=str(exc),
            error=str(exc),
            sources=[],
        )
    finally:
        _cleanup_analysis_snapshot(job_id, analysis_id)


def _start_analysis(job_id: str, analysis_id: str, paths: list[Path], settings: dict) -> None:
    threading.Thread(
        target=_run_analysis,
        args=(job_id, analysis_id, paths, settings),
        daemon=True,
        name=f"webp-analysis-{job_id[:8]}-{analysis_id[:6]}",
    ).start()


def _resume_interrupted_jobs() -> None:
    for manifest in GLOBAL.root.glob("*/job.json"):
        try:
            job = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(job.get("status", "")) != "interrupted":
            continue
        settings = job.get("settings")
        sources = list(job.get("sources", []))
        if not isinstance(settings, dict) or not sources:
            continue
        paths = [GLOBAL.job_dir(str(job["id"])) / str(item.get("path", "")) for item in sources]
        if not all(path.is_file() for path in paths):
            continue
        GLOBAL.update(
            str(job["id"]),
            status="queued",
            message="Resuming persisted render after server restart",
            error=None,
        )
        _start_global_render(str(job["id"]), paths, settings)


def _resume_interrupted_analyses() -> None:
    for manifest in GLOBAL.root.glob("*/job.json"):
        try:
            job = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        analysis = job.get("analysis")
        if not isinstance(analysis, dict) or str(analysis.get("status", "")) not in {"queued", "running"}:
            continue
        settings = analysis.get("settings") or job.get("settings")
        job_id = str(job.get("id", ""))
        paths = _analysis_paths(job_id, analysis)
        if not isinstance(settings, dict) or not paths or not all(path.is_file() for path in paths):
            analysis_id = str(analysis.get("id") or "")
            if analysis_id:
                _analysis_update(
                    job_id,
                    analysis_id,
                    status="error",
                    message="The persisted analysis source snapshot is missing.",
                    error="The persisted analysis source snapshot is missing.",
                    sources=[],
                )
            continue
        analysis_id = str(analysis.get("id") or uuid.uuid4().hex)
        _analysis_update(
            job_id,
            analysis_id,
            status="queued",
            progress=0,
            message="Resuming persisted analysis after server restart",
            error=None,
            settings=settings,
        )
        _start_analysis(job_id, analysis_id, paths, settings)


class Handler(legacy.Handler):
    server_version = "AnimAlignWebP/6.1-temporal-jobs"

    def read_json_body(self, max_bytes: int = 1024 * 1024) -> dict:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length.") from exc
        if content_length <= 0:
            return {}
        if content_length > max_bytes:
            raise OverflowError("JSON request is too large.")
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError("Expected application/json.")
        raw = self.rfile.read(content_length)
        value = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            self.send_bytes(200, INDEX_HTML, "text/html; charset=utf-8")
            return

        if path == "/workspace-ui.js":
            self.send_bytes(200, WORKSPACE_SCRIPT, "text/javascript; charset=utf-8")
            return

        if path == "/advanced-ui.js":
            self.send_bytes(200, ADVANCED_SCRIPT, "text/javascript; charset=utf-8")
            return

        if path == "/engine-status":
            self.send_json(200, engine_status(legacy))
            return

        if path == "/job":
            job_id = query.get("id", [""])[0]
            job = GLOBAL.get(job_id)
            if not job:
                self.send_json(404, {"error": "Unknown global job."})
                return
            public = GLOBAL.public(job)
            analysis = job.get("analysis")
            if isinstance(analysis, dict):
                public["analysis"] = {
                    "id": analysis.get("id"),
                    "status": analysis.get("status"),
                    "progress": analysis.get("progress", 0),
                    "message": analysis.get("message", ""),
                }
            self.send_json(200, public)
            return

        if path == "/analysis":
            job_id = query.get("id", [""])[0]
            requested_analysis = query.get("analysis_id", [""])[0]
            job = GLOBAL.get(job_id)
            if not job:
                self.send_json(404, {"error": "Unknown global job."})
                return
            analysis = job.get("analysis")
            if not isinstance(analysis, dict):
                self.send_json(404, {"error": "No analysis exists for this job."})
                return
            if requested_analysis and str(analysis.get("id", "")) != requested_analysis:
                self.send_json(404, {"error": "That analysis run is no longer current."})
                return
            self.send_json(200, analysis)
            return

        if path == "/job/source":
            job_id = query.get("id", [""])[0]
            try:
                index = int(query.get("index", [""])[0])
            except ValueError:
                self.send_text(400, "Invalid source-frame index.")
                return
            info = GLOBAL.source_info(job_id, index)
            if not info:
                self.send_text(404, "Global job source frame not found.")
                return
            source_path, name = info
            mime = MIME_BY_SUFFIX.get(Path(name).suffix.lower(), "application/octet-stream")
            self.send_bytes(
                200,
                source_path.read_bytes(),
                mime,
                {"Content-Disposition": f'inline; filename="{Path(name).name}"'},
            )
            return

        if path == "/progress":
            job_id = query.get("id", [""])[0]
            job = GLOBAL.get(job_id)
            if job:
                self.send_json(
                    200,
                    {
                        "status": job.get("status", "draft"),
                        "progress": job.get("progress", 0),
                        "message": job.get("message", ""),
                        "error": job.get("error"),
                        "output_available": bool(GLOBAL.output_path(job_id)),
                    },
                )
                return

        if path == "/download":
            job_id = query.get("id", [""])[0]
            output = GLOBAL.output_path(job_id)
            if output:
                self.send_bytes(
                    200,
                    output.read_bytes(),
                    "image/webp",
                    {"Content-Disposition": 'attachment; filename="animation.webp"'},
                )
                return
            if GLOBAL.get(job_id):
                self.send_text(409, "This global job does not have a finished WebP yet.")
                return

        super().do_GET()

    def _parse_frame_upload(self):
        content_type, body = self.read_upload_body()
        fields, uploads_raw = legacy.parse_multipart(content_type, body)
        selected = [
            (filename, payload)
            for field, filename, payload in uploads_raw
            if field == "frames" and filename
        ]
        if not selected:
            raise ValueError("No frames were uploaded.")
        for filename, _ in selected:
            if Path(filename).suffix.lower() not in legacy.ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported image type: {filename}")
        return fields, selected

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/job/new":
            try:
                payload = self.read_json_body()
                requested = str(payload.get("job_id", "")).strip().lower() or None
                job = GLOBAL.ensure(requested)
                self.send_json(200, GLOBAL.public(job))
            except OverflowError as exc:
                self.send_json(413, {"error": str(exc)})
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})
            return

        if path == "/analyse":
            try:
                fields, selected = self._parse_frame_upload()
                settings = _settings_from_fields(fields)
                requested = str(fields.get("global_job_id", "")).strip().lower()
                job_id = requested or GLOBAL.new_id()
                GLOBAL.ensure(job_id)
                analysis_id = uuid.uuid4().hex
                paths, sources = _stage_analysis_sources(job_id, analysis_id, selected)
                GLOBAL.update(
                    job_id,
                    analysis={
                        "id": analysis_id,
                        "status": "queued",
                        "progress": 0,
                        "message": "Analysis upload received and persisted",
                        "error": None,
                        "settings": settings,
                        "sources": sources,
                        "result": None,
                    },
                )
                _start_analysis(job_id, analysis_id, paths, settings)
                self.send_json(
                    202,
                    {
                        "job_id": job_id,
                        "global_job_id": job_id,
                        "analysis_id": analysis_id,
                    },
                )
            except OverflowError as exc:
                self.send_text(413, str(exc))
            except Exception as exc:
                self.send_text(400, str(exc))
            return

        if path == "/generate":
            try:
                fields, selected = self._parse_frame_upload()
                requested = str(fields.get("global_job_id", "")).strip().lower()
                job_id = requested or GLOBAL.new_id()
                settings = _settings_from_fields(fields)
                _, paths = GLOBAL.begin_render(job_id, selected, settings)
                _start_global_render(job_id, paths, settings)
                self.send_json(202, {"job_id": job_id, "global_job_id": job_id})
            except OverflowError as exc:
                self.send_text(413, str(exc))
            except Exception as exc:
                self.send_text(400, str(exc))
            return

        super().do_POST()


def main():
    _resume_interrupted_jobs()
    _resume_interrupted_analyses()
    statuses = engine_status(legacy)
    server = ThreadingHTTPServer((legacy.HOST, legacy.PORT), Handler)
    print(f"WebP Animator persistent server listening on http://{legacy.HOST}:{legacy.PORT}")
    print(f"Local access: http://127.0.0.1:{legacy.PORT}")
    print(f"LAN access:   http://<this-machine-LAN-IP>:{legacy.PORT}")
    print(f"GPU matching: {'ready' if GPU_GEOMETRY else 'CPU fallback'}")
    for name in ("rife", "amt", "eden", "speed"):
        print(f"{name.upper():<12} {'ready' if statuses.get(name, {}).get('ready') else 'not installed'}")
    print("Global jobs:  durable with no automatic expiry")
    print("Analysis:     durable background jobs with isolated source snapshots")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
