#!/usr/bin/env python3
from __future__ import annotations

import io
import tempfile
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from anim_align_webp import (
    load_rgba,
    register_sequence,
    render_union_canvas,
    save_webp,
)

HOST = "0.0.0.0"
PORT = 18743
MAX_UPLOAD_BYTES = 250 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}

INDEX_HTML = (Path(__file__).parent / "templates" / "index.html").read_bytes()


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
    """
    Parse a multipart/form-data request with Python's standard email parser.

    Returns:
        fields: dict[str, str]
        files:  list[tuple[field_name, filename, bytes]]
    """
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


class Handler(BaseHTTPRequestHandler):
    server_version = "AnimAlignWebP/1.0"

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

    def send_text(self, status: int, text: str):
        self.send_bytes(
            status,
            text.encode("utf-8"),
            "text/plain; charset=utf-8",
        )

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self.send_bytes(
                200,
                INDEX_HTML,
                "text/html; charset=utf-8",
            )
            return

        self.send_text(404, "Not found.")

    def do_POST(self):
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
            self.send_text(
                413,
                "Upload too large. Maximum request size is 250 MB.",
            )
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

        max_shift = clamp_int(fields.get("max_shift"), 64, 0, 2000)
        duration = clamp_int(fields.get("duration"), 100, 1, 60000)
        sigma = clamp_float(fields.get("sigma"), 24.0, 0.1, 255.0)
        alpha_threshold = clamp_int(
            fields.get("alpha_threshold"), 8, 0, 255
        )
        quality = clamp_int(fields.get("quality"), 90, 0, 100)
        lossy = fields.get("lossy") == "on"

        try:
            with tempfile.TemporaryDirectory(
                prefix="anim_align_web_"
            ) as temp_dir:
                temp_dir = Path(temp_dir)
                paths = []

                # Multipart order is preserved, so this is also animation order.
                for index, (_, filename, payload) in enumerate(uploads):
                    suffix = Path(filename).suffix.lower()

                    if suffix not in ALLOWED_EXTENSIONS:
                        self.send_text(
                            400,
                            f"Unsupported image type: {filename}",
                        )
                        return

                    file_path = temp_dir / f"{index:05d}{suffix}"
                    file_path.write_bytes(payload)
                    paths.append(file_path)

                images = [load_rgba(path) for path in paths]

                positions, _ = register_sequence(
                    images,
                    axis=axis,
                    max_shift_x=max_shift,
                    max_shift_y=max_shift,
                    sigma=sigma,
                    alpha_threshold=alpha_threshold,
                    proxy_max_side=320,
                )

                frames, _, _ = render_union_canvas(images, positions)

                output_path = temp_dir / "animation.webp"
                save_webp(
                    frames,
                    output_path,
                    duration_ms=duration,
                    loop=0,
                    lossless=not lossy,
                    quality=quality,
                )

                webp = output_path.read_bytes()

            self.send_bytes(
                200,
                webp,
                "image/webp",
                {
                    "Content-Disposition":
                        'attachment; filename="animation.webp"'
                },
            )

        except Exception as exc:
            self.send_text(500, f"Generation failed: {exc}")

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} - {fmt % args}")


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Animation Frame Aligner listening on http://{HOST}:{PORT}")
    print(f"Local access: http://127.0.0.1:{PORT}")
    print(f"LAN access:   http://<this-machine-LAN-IP>:{PORT}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
