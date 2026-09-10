from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

import openai_repair


def _load_rgba(payload: bytes, label: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGBA").copy()
    except Exception as exc:
        raise ValueError(f"{label} is not a readable image.") from exc


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def install() -> None:
    import app as legacy

    if getattr(legacy.Handler, "_openai_repair_http_installed", False):
        return
    legacy.Handler._openai_repair_http_installed = True
    original_get = legacy.Handler.do_GET
    original_post = legacy.Handler.do_POST

    def do_get(handler):
        if urlparse(handler.path).path == "/openai-repair-status":
            handler.send_json(200, openai_repair.status())
            return
        return original_get(handler)

    def do_post(handler):
        if urlparse(handler.path).path != "/openai-repair":
            return original_post(handler)
        try:
            content_type, body = handler.read_upload_body()
            fields, uploads = legacy.parse_multipart(content_type, body)
            by_field: dict[str, tuple[str, bytes]] = {}
            for field, filename, payload in uploads:
                if field in {"previous", "candidate", "following", "mask"} and filename:
                    by_field[field] = (filename, payload)
            for required in ("previous", "candidate", "following"):
                if required not in by_field:
                    raise ValueError(f"Missing {required} frame.")

            previous = _load_rgba(by_field["previous"][1], "Previous frame")
            candidate = _load_rgba(by_field["candidate"][1], "Candidate frame")
            following = _load_rgba(by_field["following"][1], "Next frame")
            if previous.size != candidate.size or following.size != candidate.size:
                raise ValueError("Previous, candidate and next frames must use the same canvas size.")
            mask = _load_rgba(by_field["mask"][1], "Defect mask") if "mask" in by_field else None

            with tempfile.TemporaryDirectory(prefix="webp-openai-repair-") as temporary:
                repaired, stats = openai_repair.repair_images(
                    [previous, candidate, following],
                    [1],
                    Path(temporary),
                    masks={1: mask} if mask is not None else None,
                    loop=False,
                    model=str(fields.get("model", "") or openai_repair.DEFAULT_MODEL),
                    quality=str(fields.get("quality", "") or openai_repair.DEFAULT_QUALITY),
                    prompt=str(fields.get("prompt", "") or openai_repair.REPAIR_PROMPT),
                )
            handler.send_json(
                200,
                {
                    "image_b64": _png_b64(repaired[1]),
                    "mime": "image/png",
                    "width": repaired[1].width,
                    "height": repaired[1].height,
                    "stats": stats,
                },
            )
        except Exception as exc:
            handler.send_json(400, {"error": str(exc)})

    legacy.Handler.do_GET = do_get
    legacy.Handler.do_POST = do_post
