from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

import openai_repair_context as openai_repair
from webp_fast import save_webp_fast

ROOT = Path(__file__).resolve().parent
UI_SCRIPT = ROOT / "openai_repair_ui.js"


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


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _persist_finished_repair(job_id: str, timeline_index: int, repaired: Image.Image) -> dict:
    """Replace one canonical live-timeline frame and rebuild the finished WebP.

    Manual mode finishes deterministic processing first. The user can then paint a
    defect on any generated/live frame; that repair must become part of the durable
    job, not merely a browser preview.
    """
    app_all = sys.modules.get("app_all")
    store = getattr(app_all, "GLOBAL", None) if app_all is not None else None
    if store is None:
        raise RuntimeError("Persistent job store is unavailable.")

    job = store.get(job_id)
    if not isinstance(job, dict):
        raise ValueError("The job for this manual repair is no longer available.")
    if str(job.get("status", "")) != "done":
        raise ValueError("Finish the deterministic run before applying a manual OpenAI repair.")

    job_root = store.job_dir(job_id)
    advanced = job_root / "advanced"
    manifest_path = advanced / "live-timeline" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("The finished timeline is not available for manual repair.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = list(manifest.get("frames", []))
    if timeline_index < 0 or timeline_index >= len(frames):
        raise ValueError("The selected live frame is no longer at that timeline position.")

    output_dir = advanced / "live-timeline" / "manual-repairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timeline_index:06d}-{uuid.uuid4().hex[:10]}.png"
    repaired.convert("RGBA").save(output_path, format="PNG")

    frame = dict(frames[timeline_index])
    prior_stage = str(frame.get("stage", "Frame"))
    frame["rel"] = output_path.relative_to(advanced).as_posix()
    frame["stage"] = f"Repaired · OpenAI manual · {prior_stage}"
    prior_engine = str(frame.get("engine", "") or "generated")
    if "openai-repair" not in prior_engine.lower():
        frame["engine"] = f"{prior_engine}+openai-repair"
    frame["manual_repair"] = True
    frames[timeline_index] = frame
    manifest["frames"] = frames
    _atomic_json(manifest_path, manifest)

    images: list[Image.Image] = []
    durations: list[int] = []
    fallback_duration = int((job.get("settings") or {}).get("duration", 100) or 100)
    try:
        for item in frames:
            rel = str(item.get("rel", ""))
            path = (advanced / rel).resolve()
            path.relative_to(advanced.resolve())
            if not path.is_file():
                raise ValueError(f"Timeline frame is missing: {rel}")
            with Image.open(path) as source:
                images.append(source.convert("RGBA").copy())
            try:
                durations.append(max(1, int(round(float(item.get("duration", fallback_duration))))))
            except (TypeError, ValueError):
                durations.append(max(1, fallback_duration))

        settings = dict(job.get("settings") or {})
        temporary_webp = job_root / f".animation-manual-{uuid.uuid4().hex}.webp"
        save_webp_fast(
            images,
            temporary_webp,
            durations=durations,
            loop=0,
            lossless=not bool(settings.get("lossy", False)),
            quality=int(settings.get("quality", 90)),
        )
        os.replace(temporary_webp, job_root / "animation.webp")
    finally:
        for image in images:
            try:
                image.close()
            except Exception:
                pass
        for temporary in job_root.glob(".animation-manual-*.webp"):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    store.update(
        job_id,
        status="done",
        progress=100,
        error=None,
        message=f"Manual OpenAI repair applied to frame {timeline_index + 1}; finished WebP rebuilt.",
    )
    return {"persisted": True, "job_id": job_id, "timeline_index": timeline_index, "webp_rebuilt": True}


def install() -> None:
    import app as legacy

    if getattr(legacy.Handler, "_openai_repair_http_installed", False):
        return
    legacy.Handler._openai_repair_http_installed = True
    original_get = legacy.Handler.do_GET
    original_post = legacy.Handler.do_POST

    def do_get(handler):
        path = urlparse(handler.path).path
        if path == "/openai-repair-status":
            handler.send_json(200, openai_repair.status())
            return
        if path == "/openai-repair-ui.js":
            if not UI_SCRIPT.is_file():
                handler.send_text(404, "OpenAI repair UI is missing.")
                return
            handler.send_bytes(
                200,
                UI_SCRIPT.read_bytes(),
                "text/javascript; charset=utf-8",
                {"Cache-Control": "no-store"},
            )
            return
        return original_get(handler)

    def do_post(handler):
        if urlparse(handler.path).path != "/openai-repair":
            return original_post(handler)
        try:
            content_type, body = handler.read_upload_body()
            fields, uploads = legacy.parse_multipart(content_type, body)
            by_field: dict[str, tuple[str, bytes]] = {}
            context_uploads: list[tuple[str, bytes]] = []
            for field, filename, payload in uploads:
                if field in {"previous", "candidate", "following", "mask"} and filename:
                    by_field[field] = (filename, payload)
                elif field == "context" and filename:
                    context_uploads.append((filename, payload))
            for required in ("previous", "candidate", "following"):
                if required not in by_field:
                    raise ValueError(f"Missing {required} frame.")

            previous = _load_rgba(by_field["previous"][1], "Previous frame")
            candidate = _load_rgba(by_field["candidate"][1], "Candidate frame")
            following = _load_rgba(by_field["following"][1], "Next frame")
            if previous.size != candidate.size or following.size != candidate.size:
                raise ValueError("Previous, candidate and next frames must use the same canvas size.")
            mask = _load_rgba(by_field["mask"][1], "Defect mask") if "mask" in by_field else None

            context_frames = [
                _load_rgba(payload, f"Context frame {index + 1}")
                for index, (_, payload) in enumerate(context_uploads)
            ]
            if context_frames and any(frame.size != candidate.size for frame in context_frames):
                raise ValueError("Whole-animation context frames must use the same canvas size as the candidate.")
            if not context_frames:
                context_frames = [previous, candidate, following]
            try:
                target_index = int(str(fields.get("candidate_index", "1") or "1"))
            except ValueError:
                target_index = 1
            target_index = max(0, min(len(context_frames) - 1, target_index))

            # Explicit manual mode requires a real painted selection. Auto pipeline
            # repair does not use this HTTP endpoint and remains mask-optional.
            request_mode = str(fields.get("repair_mode", "") or "").strip().lower()
            if request_mode == "manual" and mask is None:
                raise ValueError("Manual repair requires a painted defect selection.")

            with tempfile.TemporaryDirectory(prefix="webp-openai-repair-") as temporary:
                repaired, stats = openai_repair.repair_triplet(
                    previous,
                    candidate,
                    following,
                    context_frames=context_frames,
                    target_index=target_index,
                    mask=mask,
                    model=str(fields.get("model", "") or openai_repair.DEFAULT_MODEL),
                    quality=str(fields.get("quality", "") or openai_repair.DEFAULT_QUALITY),
                    prompt=str(fields.get("prompt", "") or openai_repair.REPAIR_PROMPT),
                )

            persistence = {"persisted": False, "webp_rebuilt": False}
            job_id = str(fields.get("job_id", "") or "").strip().lower()
            timeline_text = str(fields.get("timeline_index", "") or "").strip()
            if job_id and timeline_text:
                try:
                    timeline_index = int(timeline_text)
                except ValueError as exc:
                    raise ValueError("Invalid manual-repair timeline index.") from exc
                persistence = _persist_finished_repair(job_id, timeline_index, repaired)

            handler.send_json(
                200,
                {
                    "image_b64": _png_b64(repaired),
                    "mime": "image/png",
                    "width": repaired.width,
                    "height": repaired.height,
                    "stats": stats,
                    **persistence,
                },
            )
        except Exception as exc:
            handler.send_json(400, {"error": str(exc)})

    legacy.Handler.do_GET = do_get
    legacy.Handler.do_POST = do_post
