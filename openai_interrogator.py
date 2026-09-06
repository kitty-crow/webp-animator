from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

TERMINAL_STATES = {"done", "error", "cancelled"}
ACTIVE_STATES = {"queued", "planning", "generating", "auditing", "replanning", "running"}


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _now() -> float:
    return time.time()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_effort(value: Any) -> str:
    value = str(value or "medium").lower()
    return value if value in {"none", "low", "medium", "high", "xhigh", "max"} else "medium"


def _normalise_quality(value: Any) -> str:
    value = str(value or "medium").lower()
    return value if value in {"low", "medium", "high"} else "medium"


def _balanced_targets(count: int) -> list[float]:
    targets = [index / (count + 1) for index in range(1, count + 1)]
    ordered: list[float] = []

    def visit(items: list[float]) -> None:
        if not items:
            return
        middle = len(items) // 2
        ordered.append(items[middle])
        visit(items[:middle])
        visit(items[middle + 1 :])

    visit(targets)
    return ordered


def _fraction_key(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def _image_to_png(payload: bytes) -> tuple[bytes, tuple[int, int], bool]:
    with Image.open(io.BytesIO(payload)) as source:
        frame = source.convert("RGBA")
        transparency = frame.getchannel("A").getextrema()[0] < 255
        output = io.BytesIO()
        frame.save(output, format="PNG")
        return output.getvalue(), frame.size, transparency


def make_contact_sheet(frame_paths: list[Path], labels: list[str], output: Path, max_frames: int = 24) -> Path:
    if not frame_paths:
        raise ValueError("No frames for contact sheet.")

    if len(frame_paths) > max_frames:
        chosen = sorted({round(i * (len(frame_paths) - 1) / (max_frames - 1)) for i in range(max_frames)})
        frame_paths = [frame_paths[i] for i in chosen]
        labels = [labels[i] for i in chosen]

    thumb_w, thumb_h = 220, 220
    label_h = 28
    cols = min(6, max(1, len(frame_paths)))
    rows = (len(frame_paths) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * thumb_w, rows * (thumb_h + label_h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sheet)

    for index, (path, label) in enumerate(zip(frame_paths, labels)):
        with Image.open(path) as source:
            frame = source.convert("RGBA")
            frame.thumbnail((thumb_w - 12, thumb_h - 12), Image.Resampling.LANCZOS)
            x = (index % cols) * thumb_w
            y = (index // cols) * (thumb_h + label_h)
            px = x + (thumb_w - frame.width) // 2
            py = y + (thumb_h - frame.height) // 2
            sheet.alpha_composite(frame, (px, py))
            draw.text((x + 6, y + thumb_h + 4), str(label), fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    return output


class BridgeClient:
    def __init__(self, root: Path, port: int = 18744):
        self.root = root
        self.port = int(os.environ.get("OPENAI_BRIDGE_PORT", port))
        self.host = os.environ.get("OPENAI_BRIDGE_HOST", "127.0.0.1")
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.log_path = root / ".openai-jobs" / "bridge.log"

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        for key, value in _read_env_file(self.root / ".env").items():
            env.setdefault(key, value)
        env["OPENAI_BRIDGE_HOST"] = self.host
        env["OPENAI_BRIDGE_PORT"] = str(self.port)
        return env

    def status(self) -> dict[str, Any]:
        env = self._environment()
        schema = self.root / "vendor" / "openai-schema" / "dist" / "openaiSchema.js"
        runtime = os.environ.get("OPENAI_BRIDGE_RUNTIME") or shutil.which("bun") or shutil.which("node")
        health: dict[str, Any] | None = None
        try:
            health = self.request("/health", None, method="GET", ensure=False)
        except Exception:
            pass
        return {
            "key_configured": bool(env.get("OPENAI_API_KEY")),
            "schema_built": schema.is_file(),
            "runtime": runtime,
            "bridge_running": bool(health and health.get("ok")),
            "bridge": health,
        }

    def ensure_started(self) -> None:
        try:
            health = self.request("/health", None, method="GET", ensure=False)
            if health.get("ok"):
                return
        except Exception:
            pass

        with self.lock:
            try:
                health = self.request("/health", None, method="GET", ensure=False)
                if health.get("ok"):
                    return
            except Exception:
                pass

            env = self._environment()
            if not env.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is missing. Put it in the repository .env file.")

            schema = self.root / "vendor" / "openai-schema" / "dist" / "openaiSchema.js"
            if not schema.is_file():
                raise RuntimeError("openai-schema is not built. Run `python setup_openai.py` first.")

            configured = os.environ.get("OPENAI_BRIDGE_RUNTIME")
            runtime = configured or shutil.which("bun") or shutil.which("node")
            if not runtime:
                raise RuntimeError("Neither Bun nor Node.js was found for the OpenAI bridge.")

            server = self.root / "openai_bridge" / "server.mjs"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            log = self.log_path.open("a", encoding="utf-8")
            self.process = subprocess.Popen(
                [runtime, str(server)],
                cwd=str(self.root),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

            deadline = time.time() + 12
            last_error = ""
            while time.time() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"OpenAI bridge exited with code {self.process.returncode}. See {self.log_path}.")
                try:
                    health = self.request("/health", None, method="GET", ensure=False)
                    if health.get("ok"):
                        return
                except Exception as exc:
                    last_error = str(exc)
                time.sleep(0.2)
            raise RuntimeError(f"OpenAI bridge did not become ready: {last_error}")

    def request(self, path: str, payload: dict[str, Any] | None, *, method: str = "POST", ensure: bool = True, timeout: int = 300) -> dict[str, Any]:
        if ensure:
            self.ensure_started()
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {} if data is None else {"Content-Type": "application/json"}
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI bridge request failed ({exc.code}): {body[:1600]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach OpenAI bridge: {exc}") from exc
        value = json.loads(body)
        if isinstance(value, dict) and value.get("error"):
            raise RuntimeError(str(value["error"]))
        if not isinstance(value, dict):
            raise RuntimeError("OpenAI bridge returned an invalid response.")
        return value


class OpenAIJobManager:
    def __init__(self, root: Path):
        self.root = root
        self.jobs_root = root / ".openai-jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.bridge = BridgeClient(root)
        self.locks: dict[str, threading.RLock] = {}
        self.master_lock = threading.Lock()
        self._mark_interrupted_jobs()

    def _job_dir(self, job_id: str) -> Path:
        if not job_id or any(ch not in "0123456789abcdef" for ch in job_id.lower()):
            raise ValueError("Invalid job id.")
        return self.jobs_root / job_id

    def _manifest_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _lock_for(self, job_id: str) -> threading.RLock:
        with self.master_lock:
            return self.locks.setdefault(job_id, threading.RLock())

    def _mark_interrupted_jobs(self) -> None:
        for manifest in self.jobs_root.glob("*/job.json"):
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
                if value.get("status") in ACTIVE_STATES:
                    value["status"] = "interrupted"
                    value["message"] = "Server restarted while this job was running. Resume it to continue."
                    value["updated"] = _now()
                    _atomic_json(manifest, value)
            except Exception:
                continue

    def get(self, job_id: str) -> dict[str, Any] | None:
        try:
            path = self._manifest_path(job_id)
        except ValueError:
            return None
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save(self, job: dict[str, Any]) -> None:
        job["updated"] = _now()
        _atomic_json(self._manifest_path(str(job["id"])), job)

    def _update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock_for(job_id):
            job = self.get(job_id)
            if not job:
                raise KeyError(job_id)
            job.update(changes)
            self._save(job)
            return job

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        attempts = []
        for item in job.get("attempts", []):
            attempts.append({
                "target_fraction": item.get("target_fraction"),
                "attempt": item.get("attempt"),
                "accepted": item.get("accepted"),
                "audit": item.get("audit"),
                "plan": item.get("plan"),
                "estimated_cost_usd": item.get("estimated_cost_usd"),
                "actual_cost_usd": item.get("actual_cost_usd"),
            })
        accepted = [
            {
                "target_fraction": item.get("target_fraction"),
                "result_url": f"/openai/result?id={job['id']}&fraction={item.get('target_fraction')}",
            }
            for item in sorted(job.get("accepted", []), key=lambda value: float(value.get("target_fraction", 0)))
        ]
        return {
            "id": job.get("id"),
            "status": job.get("status"),
            "stage": job.get("stage"),
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "error": job.get("error"),
            "created": job.get("created"),
            "updated": job.get("updated"),
            "spent_usd": job.get("spent_usd", 0),
            "estimated_next_usd": job.get("estimated_next_usd", 0),
            "request": job.get("request", {}),
            "attempts": attempts,
            "accepted": accepted,
            "pending_target": job.get("pending_target"),
        }

    def create(self, uploads: list[tuple[str, bytes]], request: dict[str, Any]) -> dict[str, Any]:
        if len(uploads) < 2:
            raise ValueError("OpenAI interpolation needs at least two frames.")
        job_id = uuid.uuid4().hex
        job_dir = self._job_dir(job_id)
        source_dir = job_dir / "source"
        source_dir.mkdir(parents=True)

        sources: list[dict[str, Any]] = []
        first_size: tuple[int, int] | None = None
        any_transparency = False
        for index, (name, payload) in enumerate(uploads):
            png, size, transparent = _image_to_png(payload)
            if first_size is None:
                first_size = size
            path = source_dir / f"{index:06d}.png"
            path.write_bytes(png)
            any_transparency = any_transparency or transparent
            sources.append({"index": index, "name": name, "path": str(path.relative_to(job_dir)), "width": size[0], "height": size[1]})

        normal = self._normalise_request(request, len(sources))
        if first_size:
            normal["width"], normal["height"] = first_size
        normal["transparent"] = any_transparency

        count = 1 if normal["mode"] == "single" else normal["count"]
        targets = _balanced_targets(count) if normal["mode"] in {"single", "fixed"} else [0.5]
        job: dict[str, Any] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "OpenAI interpolation job queued",
            "created": _now(),
            "updated": _now(),
            "request": normal,
            "sources": sources,
            "targets_remaining": targets,
            "accepted": [],
            "attempts": [],
            "spent_usd": 0.0,
            "estimated_next_usd": 0.0,
            "pending_target": None,
        }
        self._save(job)
        self._start(job_id)
        return self.public(job)

    def _normalise_request(self, request: dict[str, Any], frame_count: int) -> dict[str, Any]:
        mode = str(request.get("mode", "single")).lower()
        if mode not in {"single", "fixed", "auto"}:
            mode = "single"
        sequence_mode = "loop" if str(request.get("sequence_mode", "open")).lower() == "loop" else "open"
        left = int(request.get("left_index", 0))
        right = int(request.get("right_index", 1))
        left = max(0, min(frame_count - 1, left))
        right = max(0, min(frame_count - 1, right))
        if left == right:
            raise ValueError("Choose two different anchor frames.")
        loop_closure = bool(request.get("loop_closure"))
        if loop_closure and sequence_mode != "loop":
            raise ValueError("Loop closure requires sequence_mode=loop.")
        if not loop_closure and left >= right:
            raise ValueError("The earlier anchor must come before the later anchor unless this is loop closure.")
        if loop_closure and not (left == frame_count - 1 and right == 0):
            raise ValueError("Loop closure must use the last frame as the earlier anchor and frame 1 as the later anchor.")

        count = max(1, min(16, int(request.get("count", 1))))
        max_frames = max(1, min(32, int(request.get("max_frames", max(4, count)))))
        max_retries = max(0, min(4, int(request.get("auto_retries", 1))))
        max_spend = max(0.0, min(1000.0, _safe_float(request.get("max_spend_usd"), 0.0)))
        min_benefit = max(0, min(100, int(request.get("min_benefit_score", 35))))

        return {
            "mode": mode,
            "sequence_mode": sequence_mode,
            "gap_type": "loop_closure" if loop_closure else "interior",
            "loop_closure": loop_closure,
            "left_index": left,
            "right_index": right,
            "count": count,
            "max_frames": max_frames,
            "min_benefit_score": min_benefit,
            "auto_retries": max_retries,
            "max_spend_usd": max_spend,
            "user_instruction": str(request.get("user_instruction", "")).strip(),
            "planner_model": str(request.get("planner_model", "gpt-5.6-luna")).strip() or "gpt-5.6-luna",
            "planner_effort": _normalise_effort(request.get("planner_effort")),
            "image_model": str(request.get("image_model", "gpt-image-2")).strip() or "gpt-image-2",
            "image_quality": _normalise_quality(request.get("image_quality")),
            "auditor_model": str(request.get("auditor_model", "gpt-5.6-luna")).strip() or "gpt-5.6-luna",
            "auditor_effort": _normalise_effort(request.get("auditor_effort")),
            "whole_sequence_context": request.get("whole_sequence_context", True) is not False,
            "rife_finish_multiplier": max(1, min(8, int(request.get("rife_finish_multiplier", 1)))),
        }

    def _start(self, job_id: str) -> None:
        thread = threading.Thread(target=self._work, args=(job_id,), daemon=True, name=f"openai-job-{job_id[:8]}")
        thread.start()

    def resume(self, job_id: str, feedback: str = "", max_spend_usd: float | None = None) -> dict[str, Any]:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.get("status") in ACTIVE_STATES:
            return self.public(job)
        if job.get("status") == "done":
            return self.public(job)
        request = dict(job.get("request", {}))
        if feedback.strip():
            request["user_feedback"] = feedback.strip()
        if max_spend_usd is not None:
            request["max_spend_usd"] = max(0.0, float(max_spend_usd))
        job["request"] = request
        job["status"] = "queued"
        job["stage"] = "queued"
        job["error"] = None
        job["message"] = "Job queued to continue with previous attempts as context"
        self._save(job)
        self._start(job_id)
        return self.public(job)

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job and job.get("status") in ACTIVE_STATES:
            raise RuntimeError("Cannot delete a running job.")
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

    def result_path(self, job_id: str, fraction: float) -> Path | None:
        job = self.get(job_id)
        if not job:
            return None
        key = _fraction_key(fraction)
        for item in job.get("accepted", []):
            if _fraction_key(float(item.get("target_fraction", -1))) == key:
                path = self._job_dir(job_id) / str(item["path"])
                return path if path.is_file() else None
        return None

    def estimate(self, request: dict[str, Any], frame_count: int, width: int, height: int) -> dict[str, Any]:
        normal = self._normalise_request(request, frame_count)
        attempts_per_frame = 1 + normal["auto_retries"]
        frames = 1 if normal["mode"] == "single" else normal["count"]
        if normal["mode"] == "auto":
            frames = normal["max_frames"]
        payload = {
            "planner_model": normal["planner_model"],
            "auditor_model": normal["auditor_model"],
            "image_quality": normal["image_quality"],
            "width": width,
            "height": height,
            "reference_count": 3,
            "context_count": 1 if normal["whole_sequence_context"] else 0,
            "attempts": attempts_per_frame * frames,
        }
        estimate = self.bridge.request("/estimate", payload)
        estimate["frames"] = frames
        estimate["attempts_per_frame"] = attempts_per_frame
        return estimate

    def _source_path(self, job: dict[str, Any], index: int) -> Path:
        source = job["sources"][index]
        return self._job_dir(job["id"]) / source["path"]

    def _b64(self, path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _context_sheet(self, job: dict[str, Any]) -> Path | None:
        if not job["request"].get("whole_sequence_context", True):
            return None
        paths = [self._source_path(job, index) for index in range(len(job["sources"]))]
        labels = [f"Frame {index + 1}" for index in range(len(paths))]
        for accepted in sorted(job.get("accepted", []), key=lambda item: float(item["target_fraction"])):
            paths.append(self._job_dir(job["id"]) / accepted["path"])
            labels.append(f"Accepted gap frame t={float(accepted['target_fraction']):.3f}")
        return make_contact_sheet(paths, labels, self._job_dir(job["id"]) / "context" / "sequence.png")

    def _anchor_paths(self, job: dict[str, Any], target: float) -> tuple[float, Path, float, Path]:
        request = job["request"]
        anchors: list[tuple[float, Path]] = [
            (0.0, self._source_path(job, int(request["left_index"]))),
            (1.0, self._source_path(job, int(request["right_index"]))),
        ]
        for item in job.get("accepted", []):
            anchors.append((float(item["target_fraction"]), self._job_dir(job["id"]) / item["path"]))
        anchors.sort(key=lambda item: item[0])
        left = anchors[0]
        right = anchors[-1]
        for item in anchors:
            if item[0] < target:
                left = item
            elif item[0] > target:
                right = item
                break
        return left[0], left[1], right[0], right[1]

    def _previous_attempt(self, job: dict[str, Any], target: float) -> dict[str, Any] | None:
        key = _fraction_key(target)
        matches = [item for item in job.get("attempts", []) if _fraction_key(float(item.get("target_fraction", -1))) == key]
        return matches[-1] if matches else None

    def _bridge_cost(self, request: dict[str, Any], planner_usage: Any, image_usage: Any, auditor_usage: Any) -> float:
        value = self.bridge.request("/cost", {
            "planner_model": request["planner_model"],
            "auditor_model": request["auditor_model"],
            "planner_usage": planner_usage,
            "image_usage": image_usage,
            "auditor_usage": auditor_usage,
        })
        return float(value.get("planner", 0)) + float(value.get("generation", 0)) + float(value.get("auditor", 0))

    def _estimate_attempt(self, job: dict[str, Any]) -> float:
        request = job["request"]
        value = self.bridge.request("/estimate", {
            "planner_model": request["planner_model"],
            "auditor_model": request["auditor_model"],
            "image_quality": request["image_quality"],
            "width": request.get("width", 1024),
            "height": request.get("height", 1024),
            "reference_count": 3,
            "context_count": 1 if request.get("whole_sequence_context", True) else 0,
            "attempts": 1,
        })
        return float(value.get("one_attempt", 0))

    def _plan_payload(self, job: dict[str, Any], target: float, left_t: float, left_path: Path, right_t: float, right_path: Path, previous: dict[str, Any] | None = None) -> dict[str, Any]:
        request = job["request"]
        local_fraction = (target - left_t) / max(1e-9, right_t - left_t)
        context = []
        sheet = self._context_sheet(job)
        if sheet:
            context.append({"label": "Whole known animation timeline", "image_b64": self._b64(sheet), "mime": "image/png"})
        payload: dict[str, Any] = {
            "planner_model": request["planner_model"],
            "planner_effort": request["planner_effort"],
            "sequence_mode": request["sequence_mode"],
            "gap_type": request["gap_type"],
            "target_fraction": local_fraction,
            "global_fraction": target,
            "user_instruction": request.get("user_instruction", ""),
            "user_feedback": request.get("user_feedback", ""),
            "frame_a_b64": self._b64(left_path),
            "frame_a_mime": "image/png",
            "frame_b_b64": self._b64(right_path),
            "frame_b_mime": "image/png",
            "context_frames": context,
        }
        if previous:
            payload["previous_plan"] = previous.get("plan")
            payload["previous_audit"] = previous.get("audit")
            previous_path = self._job_dir(job["id"]) / str(previous.get("candidate_path", ""))
            if previous_path.is_file():
                payload["previous_attempt_b64"] = self._b64(previous_path)
                payload["previous_attempt_mime"] = "image/png"
        return payload

    def _generation_prompt(self, plan: dict[str, Any], request: dict[str, Any], local_fraction: float) -> str:
        return (
            "Generate exactly one missing animation frame between the supplied temporal reference frames. "
            f"The requested position is {local_fraction:.6f} of the way from the earlier immediate anchor to the later immediate anchor. "
            "This is a generic animation utility: infer the actual visual subject from the supplied images and do not introduce unrelated content. "
            "Preserve the canvas composition, visual identity, style, geometry, colours, lighting, texture, transparency/background behaviour, and all regions that do not need to change. "
            "Change only the minimum visual regions necessary to realise the temporal state described by the interpolation plan. "
            "Do not add decorative detail, redesign existing content, alter the camera/framing, crop the image, add text, or invent objects unless such a change is unavoidably implied by the surrounding frames. "
            "Use both temporal anchors as constraints rather than treating either as a loose style reference. "
            f"User intent: {request.get('user_instruction') or '(none; infer motion only from the animation)'}\n"
            f"Validated interpolation plan: {json.dumps(plan, ensure_ascii=False)}"
        )

    def _attempt(self, job_id: str, target: float, preplan: dict[str, Any] | None = None) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        request = job["request"]
        left_t, left_path, right_t, right_path = self._anchor_paths(job, target)
        local_fraction = (target - left_t) / max(1e-9, right_t - left_t)
        previous = self._previous_attempt(job, target)
        attempt_number = 1 + sum(1 for item in job.get("attempts", []) if _fraction_key(float(item.get("target_fraction", -1))) == _fraction_key(target))

        estimate = self._estimate_attempt(job)
        spent = float(job.get("spent_usd", 0.0))
        max_spend = float(request.get("max_spend_usd", 0.0))
        self._update(job_id, estimated_next_usd=estimate)
        if max_spend > 0 and spent + estimate > max_spend:
            self._update(job_id, status="budget_wait", stage="budget", pending_target=target, message=f"Next attempt is estimated at ${estimate:.4f}, exceeding the ${max_spend:.2f} job budget.")
            return False

        stage = "replanning" if previous else "planning"
        self._update(job_id, status=stage, stage=stage, message=f"{stage.title()} missing frame at t={target:.3f}")
        plan_result = preplan or self.bridge.request("/plan", self._plan_payload(job, target, left_t, left_path, right_t, right_path, previous))
        plan = plan_result["plan"]
        if not plan.get("valid_pair", True):
            raise RuntimeError("Planner rejected this anchor pair as unsuitable for interpolation.")

        generation_instruction = self._generation_prompt(plan, request, local_fraction)
        references = [
            {"image_b64": self._b64(left_path), "mime": "image/png"},
            {"image_b64": self._b64(right_path), "mime": "image/png"},
        ]
        if previous:
            previous_path = self._job_dir(job_id) / str(previous.get("candidate_path", ""))
            if previous_path.is_file():
                references.append({"image_b64": self._b64(previous_path), "mime": "image/png"})

        self._update(job_id, status="generating", stage="generating", message=f"Generating attempt {attempt_number} for t={target:.3f}")
        size = f"{int(request.get('width', 1024))}x{int(request.get('height', 1024))}"
        generated = self.bridge.request("/generate", {
            "image_model": request["image_model"],
            "image_quality": request["image_quality"],
            "generation_instruction": generation_instruction,
            "references": references,
            "size": size,
            "transparent": bool(request.get("transparent", False)),
        }, timeout=600)

        target_dir = self._job_dir(job_id) / "attempts" / f"target-{_fraction_key(target).replace('.', '_')}"
        target_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = target_dir / f"attempt-{attempt_number:02d}.png"
        candidate_path.write_bytes(base64.b64decode(generated["image_b64"]))
        expected_size = (int(request.get("width", 1024)), int(request.get("height", 1024)))
        with Image.open(candidate_path) as candidate_source:
            candidate = candidate_source.convert("RGBA")
            if candidate.size != expected_size:
                candidate = candidate.resize(expected_size, Image.Resampling.LANCZOS)
            candidate.save(candidate_path, format="PNG")

        self._update(job_id, status="auditing", stage="auditing", message=f"Auditing attempt {attempt_number} for t={target:.3f}")
        context = []
        sheet = self._context_sheet(job)
        if sheet:
            context.append({"label": "Whole known animation timeline", "image_b64": self._b64(sheet), "mime": "image/png"})
        audit_result = self.bridge.request("/audit", {
            "auditor_model": request["auditor_model"],
            "auditor_effort": request["auditor_effort"],
            "sequence_mode": request["sequence_mode"],
            "gap_type": request["gap_type"],
            "target_fraction": local_fraction,
            "global_fraction": target,
            "user_instruction": request.get("user_instruction", ""),
            "plan": plan,
            "frame_a_b64": self._b64(left_path),
            "frame_a_mime": "image/png",
            "candidate_b64": self._b64(candidate_path),
            "candidate_mime": "image/png",
            "frame_b_b64": self._b64(right_path),
            "frame_b_mime": "image/png",
            "context_frames": context,
        })
        audit = audit_result["audit"]

        actual = self._bridge_cost(request, plan_result.get("usage"), generated.get("usage"), audit_result.get("usage"))
        if actual <= 0:
            actual = estimate

        job = self.get(job_id)
        if not job:
            return False
        attempt = {
            "target_fraction": target,
            "attempt": attempt_number,
            "plan": plan,
            "audit": audit,
            "candidate_path": str(candidate_path.relative_to(self._job_dir(job_id))),
            "accepted": bool(audit.get("acceptable")),
            "estimated_cost_usd": estimate,
            "actual_cost_usd": actual,
        }
        job.setdefault("attempts", []).append(attempt)
        job["spent_usd"] = float(job.get("spent_usd", 0.0)) + actual
        job["estimated_next_usd"] = 0.0

        if audit.get("acceptable"):
            accepted_dir = self._job_dir(job_id) / "accepted"
            accepted_dir.mkdir(parents=True, exist_ok=True)
            accepted_path = accepted_dir / f"t-{_fraction_key(target).replace('.', '_')}.png"
            shutil.copyfile(candidate_path, accepted_path)
            job.setdefault("accepted", []).append({"target_fraction": target, "path": str(accepted_path.relative_to(self._job_dir(job_id)))})
            job["pending_target"] = None
            self._save(job)
            return True

        self._save(job)
        return False

    def _next_auto(self, job: dict[str, Any]) -> tuple[float | None, dict[str, Any] | None]:
        request = job["request"]
        if len(job.get("accepted", [])) >= int(request["max_frames"]):
            return None, None
        anchors = [0.0, 1.0] + [float(item["target_fraction"]) for item in job.get("accepted", [])]
        anchors = sorted(set(anchors))
        intervals = sorted(((anchors[index + 1] - anchors[index], anchors[index], anchors[index + 1]) for index in range(len(anchors) - 1)), reverse=True)
        if not intervals:
            return None, None
        _, left_t, right_t = intervals[0]
        target = (left_t + right_t) / 2
        _, left_path, _, right_path = self._anchor_paths(job, target)
        plan_result = self.bridge.request("/plan", self._plan_payload(job, target, left_t, left_path, right_t, right_path, None))
        plan = plan_result.get("plan", {})
        if not plan.get("additional_frame_worthwhile") or int(plan.get("benefit_score", 0)) < int(request["min_benefit_score"]):
            decision_cost = self._bridge_cost(request, plan_result.get("usage"), None, None)
            job = self.get(job["id"]) or job
            job["spent_usd"] = float(job.get("spent_usd", 0.0)) + decision_cost
            self._save(job)
            return None, None
        return target, plan_result

    def _work(self, job_id: str) -> None:
        worker_lock = self._lock_for(job_id)
        if not worker_lock.acquire(blocking=False):
            return
        try:
            job = self.get(job_id)
            if not job:
                return
            self._update(job_id, status="running", stage="starting", message="Starting OpenAI interpolation worker")

            while True:
                job = self.get(job_id)
                if not job:
                    return
                request = job["request"]
                remaining = list(job.get("targets_remaining", []))

                if not remaining:
                    if request["mode"] == "auto":
                        self._update(job_id, status="planning", stage="planning", message="Checking whether another frame would materially improve the animation")
                        job = self.get(job_id) or job
                        target, preplan = self._next_auto(job)
                        if target is None:
                            self._finish(job_id)
                            return
                        job = self.get(job_id) or job
                        job["targets_remaining"] = [target]
                        job.setdefault("preplans", {})[_fraction_key(target)] = preplan
                        self._save(job)
                        remaining = [target]
                    else:
                        self._finish(job_id)
                        return

                target = float(remaining[0])
                job = self.get(job_id) or job
                preplan = (job.get("preplans") or {}).pop(_fraction_key(target), None)
                if preplan is not None:
                    self._save(job)

                accepted = False
                max_attempts = 1 + int(request.get("auto_retries", 0))
                existing = sum(1 for item in job.get("attempts", []) if _fraction_key(float(item.get("target_fraction", -1))) == _fraction_key(target))
                attempts_left = max(1, max_attempts - existing)
                for _ in range(attempts_left):
                    accepted = self._attempt(job_id, target, preplan=preplan)
                    preplan = None
                    current = self.get(job_id)
                    if not current:
                        return
                    if current.get("status") == "budget_wait":
                        return
                    if accepted:
                        break

                job = self.get(job_id)
                if not job:
                    return
                if not accepted:
                    self._update(
                        job_id,
                        status="needs_review",
                        stage="review",
                        pending_target=target,
                        message="The auditor rejected the latest frame. Add feedback and try again, or increase automatic retries.",
                    )
                    return

                job["targets_remaining"] = [float(value) for value in job.get("targets_remaining", []) if _fraction_key(float(value)) != _fraction_key(target)]
                completed = len(job.get("accepted", []))
                total_hint = max(completed + len(job["targets_remaining"]), 1)
                job["progress"] = min(95, round(completed / total_hint * 90))
                job["message"] = f"Accepted generated frame {completed} at t={target:.3f}"
                job["status"] = "running"
                job["stage"] = "continuing"
                job["request"].pop("user_feedback", None)
                self._save(job)

        except Exception as exc:
            try:
                self._update(job_id, status="error", stage="error", error=str(exc), message=str(exc))
            except Exception:
                pass
        finally:
            worker_lock.release()

    def _finish(self, job_id: str) -> None:
        job = self.get(job_id)
        if not job:
            return
        accepted = sorted(job.get("accepted", []), key=lambda item: float(item["target_fraction"]))
        job["accepted"] = accepted
        job["status"] = "done"
        job["stage"] = "done"
        job["progress"] = 100
        job["pending_target"] = None
        job["estimated_next_usd"] = 0.0
        job["message"] = f"OpenAI interpolation complete: {len(accepted)} accepted frame{'s' if len(accepted) != 1 else ''}."
        self._save(job)

    def cleanup(self, completed_ttl: int = 24 * 3600, review_ttl: int = 7 * 24 * 3600) -> None:
        now = _now()
        for manifest in self.jobs_root.glob("*/job.json"):
            try:
                job = json.loads(manifest.read_text(encoding="utf-8"))
                status = job.get("status")
                if status in ACTIVE_STATES:
                    continue
                age = now - float(job.get("updated", job.get("created", now)))
                ttl = review_ttl if status in {"needs_review", "budget_wait", "interrupted"} else completed_ttl
                if age > ttl:
                    shutil.rmtree(manifest.parent, ignore_errors=True)
            except Exception:
                continue
