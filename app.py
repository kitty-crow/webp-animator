#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from anim_align_webp import (
    load_rgba,
    register_sequence,
    render_union_canvas,
    save_webp,
)

ROOT = Path(__file__).resolve().parent
HOST = "0.0.0.0"
PORT = 18743
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
JOB_TTL_SECONDS = 60 * 60

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}

INDEX_HTML = (ROOT / "templates" / "index.html").read_bytes()
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def clamp_int(value, default_value, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default_value
    return max(minimum, min(maximum, value))


def clamp_float(value, default_value, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default_value
    return max(minimum, min(maximum, value))


def parse_multipart(content_type: str, body: bytes):
    message = BytesParser(policy=default).parsebytes(
        (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n"
            "\r\n"
        ).encode("utf-8")
        + body
    )

    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data")

    fields = {}
    files = []

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue

        name = part.get_param("name", header="Content-Disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""

        if filename is not None:
            files.append((name or "", filename, payload))
        elif name:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, errors="replace")

    return fields, files


def set_job(job_id: str, **changes):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job.update(changes)


def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def clean_old_jobs():
    now = time.time()
    stale = []
    with JOBS_LOCK:
        for job_id, job in JOBS.items():
            if now - job.get("created", now) > JOB_TTL_SECONDS:
                stale.append((job_id, job.get("job_dir")))
        for job_id, _ in stale:
            JOBS.pop(job_id, None)

    for _, directory in stale:
        if directory:
            shutil.rmtree(directory, ignore_errors=True)


def rife_paths():
    rife_python = Path(
        os.environ.get("RIFE_PYTHON", ROOT / ".rife-venv" / "bin" / "python")
    ).expanduser()
    rife_dir = Path(
        os.environ.get("RIFE_DIR", ROOT / "third_party" / "Practical-RIFE")
    ).expanduser()
    model_dir = Path(
        os.environ.get("RIFE_MODEL_DIR", rife_dir / "train_log")
    ).expanduser()

    ready = (
        rife_python.is_file()
        and rife_dir.is_dir()
        and model_dir.is_dir()
        and (model_dir / "flownet.pkl").is_file()
    )
    return ready, rife_python, rife_dir, model_dir


def run_rife(
    job_id: str,
    input_dir: Path,
    output_dir: Path,
    multiplier: int,
):
    ready, rife_python, rife_dir, model_dir = rife_paths()
    if not ready:
        raise RuntimeError(
            "RIFE is not installed. Run `python setup_rife.py` in the project "
            "directory, then restart the web server."
        )

    command = [
        str(rife_python),
        str(ROOT / "rife_worker.py"),
        "--rife-dir", str(rife_dir),
        "--model-dir", str(model_dir),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--multi", str(multiplier),
    ]

    process = subprocess.Popen(
        command,
        cwd=str(rife_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_tail = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if line.startswith("PROGRESS "):
            try:
                _, current, total = line.split()
                current_i = int(current)
                total_i = max(1, int(total))
                fraction = min(1.0, current_i / total_i)
                set_job(
                    job_id,
                    progress=45 + round(fraction * 40),
                    message=f"RIFE interpolation {current_i}/{total_i}",
                )
            except (ValueError, IndexError):
                pass
        elif line:
            log_tail.append(line)
            log_tail = log_tail[-20:]

    return_code = process.wait()
    if return_code != 0:
        details = "\n".join(log_tail[-8:]) or "No RIFE worker output."
        raise RuntimeError(f"RIFE interpolation failed:\n{details}")


def process_job(job_id: str, paths: list[Path], settings: dict):
    job = get_job(job_id)
    if not job:
        return

    job_dir = Path(job["job_dir"])

    try:
        set_job(job_id, progress=12, message="Loading frames")
        images = [load_rgba(path) for path in paths]

        set_job(job_id, progress=22, message="Registering frame positions")
        positions, _ = register_sequence(
            images,
            axis=settings["axis"],
            max_shift_x=settings["max_shift"],
            max_shift_y=settings["max_shift"],
            sigma=settings["sigma"],
            alpha_threshold=settings["alpha_threshold"],
            proxy_max_side=320,
        )

        set_job(job_id, progress=38, message="Rendering aligned frames")
        frames, _, _ = render_union_canvas(images, positions)
        duration = settings["duration"]

        multiplier = settings["rife_multiplier"]
        if multiplier > 1:
            aligned_dir = job_dir / "aligned"
            interpolated_dir = job_dir / "interpolated"
            aligned_dir.mkdir(parents=True, exist_ok=True)
            interpolated_dir.mkdir(parents=True, exist_ok=True)

            for index, frame in enumerate(frames):
                frame.save(aligned_dir / f"{index:06d}.png")

            set_job(job_id, progress=45, message="Starting RIFE interpolation")
            run_rife(job_id, aligned_dir, interpolated_dir, multiplier)

            result_paths = sorted(
                interpolated_dir.glob("*.png"),
                key=lambda path: int(path.stem),
            )
            if not result_paths:
                raise RuntimeError("RIFE produced no output frames.")
            frames = [load_rgba(path) for path in result_paths]
            duration = max(1, round(duration / multiplier))

        set_job(job_id, progress=90, message="Encoding animated WebP")
        output_path = job_dir / "animation.webp"
        save_webp(
            frames,
            output_path,
            duration_ms=duration,
            loop=0,
            lossless=not settings["lossy"],
            quality=settings["quality"],
        )

        set_job(
            job_id,
            progress=100,
            status="done",
            message=f"Ready: {len(frames)} frames at {duration} ms/frame",
            output_path=str(output_path),
        )
    except Exception as exc:
        set_job(
            job_id,
            status="error",
            message=str(exc),
            error=str(exc),
        )


class Handler(BaseHTTPRequestHandler):
    server_version = "AnimAlignWebP/2.0"

    def send_bytes(
        self,
        status: int,
        data: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status: int, payload: dict):
        self.send_bytes(
            status,
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def send_text(self, status: int, text: str):
        self.send_bytes(
            status,
            text.encode("utf-8"),
            "text/plain; charset=utf-8",
        )

    def do_GET(self):
        clean_old_jobs()
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            self.send_bytes(200, INDEX_HTML, "text/html; charset=utf-8")
            return

        if path == "/rife-status":
            ready, rife_python, rife_dir, model_dir = rife_paths()
            self.send_json(
                200,
                {
                    "ready": ready,
                    "python": str(rife_python),
                    "rife_dir": str(rife_dir),
                    "model_dir": str(model_dir),
                },
            )
            return

        if path == "/progress":
            job_id = query.get("id", [""])[0]
            job = get_job(job_id)
            if not job:
                self.send_json(404, {"error": "Unknown job."})
                return
            self.send_json(
                200,
                {
                    "status": job.get("status", "running"),
                    "progress": job.get("progress", 0),
                    "message": job.get("message", ""),
                    "error": job.get("error"),
                },
            )
            return

        if path == "/download":
            job_id = query.get("id", [""])[0]
            job = get_job(job_id)
            if not job:
                self.send_text(404, "Unknown job.")
                return
            if job.get("status") != "done":
                self.send_text(409, "The WebP is not ready yet.")
                return
            output_path = Path(job["output_path"])
            if not output_path.is_file():
                self.send_text(410, "The generated file is no longer available.")
                return
            self.send_bytes(
                200,
                output_path.read_bytes(),
                "image/webp",
                {"Content-Disposition": 'attachment; filename="animation.webp"'},
            )
            return

        self.send_text(404, "Not found.")

    def do_POST(self):
        clean_old_jobs()
        path = urlparse(self.path).path
        if path != "/generate":
            self.send_text(404, "Not found.")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_text(400, "Invalid Content-Length.")
            return

        if content_length <= 0:
            self.send_text(400, "Empty request.")
            return
        if content_length > MAX_UPLOAD_BYTES:
            self.send_text(413, "Upload too large. Maximum request size is 250 MB.")
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_text(400, "Expected multipart/form-data.")
            return

        try:
            body = self.rfile.read(content_length)
            fields, uploads = parse_multipart(content_type, body)
        except Exception as exc:
            self.send_text(400, f"Could not parse upload: {exc}")
            return

        uploads = [
            (field, filename, payload)
            for field, filename, payload in uploads
            if field == "frames" and filename
        ]
        if not uploads:
            self.send_text(400, "No frames were uploaded.")
            return

        axis = fields.get("axis", "xy")
        if axis not in {"x", "y", "xy", "none"}:
            axis = "xy"

        rife_multiplier = clamp_int(fields.get("rife_multiplier"), 1, 1, 8)
        if rife_multiplier not in {1, 2, 4, 8}:
            rife_multiplier = 1

        settings = {
            "axis": axis,
            "max_shift": clamp_int(fields.get("max_shift"), 64, 0, 2000),
            "duration": clamp_int(fields.get("duration"), 100, 1, 60000),
            "sigma": clamp_float(fields.get("sigma"), 24.0, 0.1, 255.0),
            "alpha_threshold": clamp_int(fields.get("alpha_threshold"), 8, 0, 255),
            "quality": clamp_int(fields.get("quality"), 90, 0, 100),
            "lossy": fields.get("lossy") == "on",
            "rife_multiplier": rife_multiplier,
        }

        job_id = uuid.uuid4().hex
        job_dir = Path(tempfile.mkdtemp(prefix=f"anim_align_{job_id[:8]}_"))
        paths = []

        try:
            for index, (_, filename, payload) in enumerate(uploads):
                suffix = Path(filename).suffix.lower()
                if suffix not in ALLOWED_EXTENSIONS:
                    raise ValueError(f"Unsupported image type: {filename}")
                file_path = job_dir / f"source_{index:06d}{suffix}"
                file_path.write_bytes(payload)
                paths.append(file_path)
        except Exception as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            self.send_text(400, str(exc))
            return

        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "running",
                "progress": 5,
                "message": "Upload received",
                "created": time.time(),
                "job_dir": str(job_dir),
            }

        thread = threading.Thread(
            target=process_job,
            args=(job_id, paths, settings),
            daemon=True,
        )
        thread.start()

        self.send_json(202, {"job_id": job_id})

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} - {fmt % args}")


def main():
    ready, _, _, _ = rife_paths()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Animation Frame Aligner listening on http://{HOST}:{PORT}")
    print(f"Local access: http://127.0.0.1:{PORT}")
    print(f"LAN access:   http://<this-machine-LAN-IP>:{PORT}")
    print(f"RIFE:         {'ready' if ready else 'not installed'}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
