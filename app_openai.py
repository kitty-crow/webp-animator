#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import app as legacy
from openai_batch import OpenAIBatchManager
from openai_interrogator import ACTIVE_STATES, OpenAIJobManager, make_contact_sheet

ROOT = Path(__file__).resolve().parent


def _atomic_manifest(path: Path, value: dict) -> None:
    """Write a manifest safely on Windows while the browser is polling it."""
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


class FullContextJobManager(OpenAIJobManager):
    """Sequence-aware OpenAI worker with Windows-safe durable manifests."""

    def __init__(self, root: Path):
        self._manifest_io_lock = threading.RLock()
        super().__init__(root)

    def get(self, job_id: str):
        try:
            path = self._manifest_path(job_id)
        except ValueError:
            return None
        with self._manifest_io_lock:
            if not path.is_file():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else None
            except Exception:
                return None

    def _save(self, job):
        job["updated"] = time.time()
        with self._manifest_io_lock:
            _atomic_manifest(self._manifest_path(str(job["id"])), job)

    def _context_sheet(self, job):
        if not job["request"].get("whole_sequence_context", True):
            return None
        paths = [self._source_path(job, index) for index in range(len(job["sources"]))]
        labels = [f"Frame {index + 1}" for index in range(len(paths))]
        for accepted in sorted(job.get("accepted", []), key=lambda item: float(item["target_fraction"])):
            paths.append(self._job_dir(job["id"]) / accepted["path"])
            labels.append(f"Accepted gap frame t={float(accepted['target_fraction']):.3f}")
        return make_contact_sheet(
            paths,
            labels,
            self._job_dir(job["id"]) / "context" / "sequence.png",
            max_frames=max(24, len(paths)),
        )


OPENAI = FullContextJobManager(ROOT)
BATCH = OpenAIBatchManager(ROOT, OPENAI)
UI_FRAGMENT = (ROOT / "openai_ui" / "fragment.html").read_text(encoding="utf-8")
UI_SCRIPT = (ROOT / "openai_ui" / "interrogator.js").read_bytes()


def enhanced_index() -> bytes:
    html = legacy.INDEX_HTML.decode("utf-8")
    marker = '      <div class="grid">'
    if marker not in html:
        raise RuntimeError("Could not find WebP Animator settings grid for OpenAI UI injection.")
    html = html.replace(marker, UI_FRAGMENT + "\n\n" + marker, 1)
    html = html.replace("</body>", '  <script src="/openai-ui.js"></script>\n</body>', 1)
    return html.encode("utf-8")


INDEX_HTML = enhanced_index()


def _fraction_key(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def attempt_path(job_id: str, fraction: float, attempt_number: int) -> Path | None:
    job = OPENAI.get(job_id)
    if not job:
        return None
    key = _fraction_key(fraction)
    for item in job.get("attempts", []):
        if (
            _fraction_key(float(item.get("target_fraction", -1))) == key
            and int(item.get("attempt", -1)) == attempt_number
        ):
            candidate = str(item.get("candidate_path", ""))
            if not candidate:
                return None
            path = OPENAI._job_dir(job_id) / candidate
            return path if path.is_file() else None
    return None


def reopen_for_retry(
    job_id: str,
    feedback: str,
    max_spend_usd: float | None,
    target_fraction: float | None,
) -> dict:
    """Requeue a stopped target, including recoverable API/infrastructure errors."""
    with OPENAI._lock_for(job_id):
        job = OPENAI.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.get("status") in ACTIVE_STATES:
            return OPENAI.public(job)
        if job.get("status") == "cancelled":
            raise ValueError("A cancelled job cannot be resumed.")

        attempts = list(job.get("attempts", []))
        if target_fraction is None:
            pending = job.get("pending_target")
            if pending is not None:
                target_fraction = float(pending)
            elif attempts:
                target_fraction = float(attempts[-1].get("target_fraction", 0.5))
            elif job.get("targets_remaining"):
                target_fraction = float(job["targets_remaining"][0])
            else:
                raise ValueError("This job has no pending or previous interpolation target to resume.")

        key = _fraction_key(float(target_fraction))
        matching = [
            item for item in attempts
            if _fraction_key(float(item.get("target_fraction", -1))) == key
        ]
        if matching:
            latest = matching[-1]
            latest["user_rejected"] = True
            latest["user_feedback"] = feedback.strip()
        elif job.get("status") not in {"interrupted", "budget_wait", "error", "needs_review"}:
            raise ValueError("That target has no previous attempt to learn from.")

        request = dict(job.get("request", {}))
        if feedback.strip():
            request["user_feedback"] = feedback.strip()
        if max_spend_usd is not None:
            request["max_spend_usd"] = max(0.0, float(max_spend_usd))
        job["request"] = request

        job["accepted"] = [
            item for item in job.get("accepted", [])
            if _fraction_key(float(item.get("target_fraction", -1))) != key
        ]
        remaining = [
            float(value) for value in job.get("targets_remaining", [])
            if _fraction_key(float(value)) != key
        ]
        job["targets_remaining"] = [float(target_fraction), *remaining]
        if isinstance(job.get("preplans"), dict):
            job["preplans"].pop(key, None)
        job["pending_target"] = float(target_fraction)
        job["status"] = "queued"
        job["stage"] = "queued"
        job["error"] = None
        job["message"] = (
            "Retry queued with the previous generated image, audit and user feedback as evidence"
            if matching
            else "Failed/interrupted target queued to continue without re-uploading the animation"
        )
        OPENAI._save(job)

    OPENAI._start(job_id)
    return OPENAI.public(job)


def estimate_request(payload: dict, frame_count: int, width: int, height: int) -> dict:
    scope = str(payload.get("scope", "pair")).lower()
    base_payload = dict(payload)
    if scope in {"range", "all"}:
        base_payload["left_index"] = 0
        base_payload["right_index"] = 1
        base_payload["loop_closure"] = False
    estimate = OPENAI.estimate(base_payload, frame_count, width, height)
    segments = OpenAIBatchManager.segment_count(payload, frame_count) if scope in {"range", "all"} else 1
    if segments > 1:
        estimate["segments"] = segments
        estimate["frames"] = int(estimate.get("frames", 1)) * segments
        estimate["attempts"] = int(estimate.get("attempts", 1)) * segments
        estimate["maximum"] = float(estimate.get("maximum", 0.0)) * segments
    else:
        estimate["segments"] = 1
    return estimate


class Handler(legacy.Handler):
    server_version = "AnimAlignWebP/3.2-openai"

    def read_json_body(self, max_bytes: int = 1024 * 1024):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length.") from exc
        if length <= 0 or length > max_bytes:
            raise ValueError("Invalid JSON request size.")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
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

        if path == "/openai-ui.js":
            self.send_bytes(200, UI_SCRIPT, "text/javascript; charset=utf-8")
            return

        if path == "/openai-status":
            OPENAI.cleanup()
            BATCH.cleanup()
            self.send_json(200, OPENAI.bridge.status())
            return

        if path == "/openai/job":
            OPENAI.cleanup()
            BATCH.cleanup()
            job_id = query.get("id", [""])[0]
            batch = BATCH.get(job_id)
            if batch:
                self.send_json(200, BATCH.public(batch))
                return
            job = OPENAI.get(job_id)
            if not job:
                self.send_json(404, {"error": "Unknown OpenAI interpolation job."})
                return
            self.send_json(200, OPENAI.public(job))
            return

        if path == "/openai/result":
            job_id = query.get("id", [""])[0]
            try:
                fraction = float(query.get("fraction", [""])[0])
            except ValueError:
                self.send_text(400, "Invalid fraction.")
                return
            result = OPENAI.result_path(job_id, fraction)
            if not result:
                self.send_text(404, "Generated frame not found.")
                return
            self.send_bytes(
                200,
                result.read_bytes(),
                "image/png",
                {"Content-Disposition": f'inline; filename="openai-{fraction:.6f}.png"'},
            )
            return

        if path == "/openai/attempt":
            job_id = query.get("id", [""])[0]
            try:
                fraction = float(query.get("fraction", [""])[0])
                attempt_number = int(query.get("attempt", [""])[0])
            except ValueError:
                self.send_text(400, "Invalid attempt selector.")
                return
            result = attempt_path(job_id, fraction, attempt_number)
            if not result:
                self.send_text(404, "OpenAI attempt image not found.")
                return
            self.send_bytes(
                200,
                result.read_bytes(),
                "image/png",
                {"Content-Disposition": f'inline; filename="openai-attempt-{attempt_number}.png"'},
            )
            return

        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/openai/estimate":
            try:
                payload = self.read_json_body()
                frame_count = max(2, int(payload.pop("frame_count", 2)))
                width = max(1, int(payload.pop("width", 1024)))
                height = max(1, int(payload.pop("height", 1024)))
                self.send_json(200, estimate_request(payload, frame_count, width, height))
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})
            return

        if path == "/openai/interpolate":
            try:
                content_type, body = self.read_upload_body()
                fields, uploads = legacy.parse_multipart(content_type, body)
                selected = [(filename, payload) for field, filename, payload in uploads if field == "frames" and filename]
                if len(selected) < 2:
                    raise ValueError("Upload at least two ordered frames for OpenAI interpolation.")
                raw_request = fields.get("openai_request", "{}")
                request = json.loads(raw_request)
                if not isinstance(request, dict):
                    raise ValueError("openai_request must be a JSON object.")
                scope = str(request.get("scope", "pair")).lower()
                if scope in {"range", "all"}:
                    job = BATCH.create(selected, request)
                else:
                    request["scope"] = "pair"
                    job = OPENAI.create(selected, request)
                self.send_json(202, job)
            except OverflowError as exc:
                self.send_json(413, {"error": str(exc)})
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})
            return

        if path == "/openai/retry":
            try:
                payload = self.read_json_body()
                job_id = str(payload.get("job_id", ""))
                feedback = str(payload.get("feedback", ""))
                max_spend = payload.get("max_spend_usd")
                max_spend_value = None if max_spend is None else float(max_spend)
                target = payload.get("target_fraction")
                target_value = None if target is None else float(target)

                batch = BATCH.get(job_id)
                if batch:
                    child_id = BATCH.current_child(job_id)
                    if child_id:
                        public_batch = BATCH.public(batch)
                        spent = float(public_batch.get("spent_usd", 0.0))
                        remaining = None
                        if max_spend_value is not None and max_spend_value > 0:
                            remaining = max(0.0, max_spend_value - spent)
                        reopen_for_retry(child_id, feedback, remaining, target_value)
                    result = BATCH.resume(job_id, max_spend_value)
                else:
                    result = reopen_for_retry(job_id, feedback, max_spend_value, target_value)
                self.send_json(202, result)
            except KeyError:
                self.send_json(404, {"error": "Unknown OpenAI interpolation job."})
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})
            return

        if path == "/openai/delete":
            try:
                payload = self.read_json_body()
                job_id = str(payload.get("job_id", ""))
                if BATCH.get(job_id):
                    BATCH.delete(job_id)
                else:
                    OPENAI.delete(job_id)
                self.send_json(200, {"deleted": True, "job_id": job_id})
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})
            return

        super().do_POST()

    def log_message(self, fmt, *args):
        super().log_message(fmt, *args)


def main():
    rife_ready, _, _, _ = legacy.rife_paths()
    server = ThreadingHTTPServer((legacy.HOST, legacy.PORT), Handler)
    openai_status = OPENAI.bridge.status()
    print(f"WebP Animator + OpenAI Interrogator listening on http://{legacy.HOST}:{legacy.PORT}")
    print(f"Local access: http://127.0.0.1:{legacy.PORT}")
    print(f"LAN access:   http://<this-machine-LAN-IP>:{legacy.PORT}")
    print(f"RIFE:         {'ready' if rife_ready else 'not installed'}")
    print(f"OpenAI:       {'configured' if openai_status['key_configured'] else 'no API key'}")
    print(f"Schema:       {'built' if openai_status['schema_built'] else 'run python setup_openai.py'}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
