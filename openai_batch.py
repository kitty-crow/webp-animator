from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from openai_interrogator import ACTIVE_STATES, OpenAIJobManager


BATCH_ACTIVE_STATES = {"queued", "running", "waiting_child"}
BATCH_STOPPED_STATES = {"done", "error", "needs_review", "budget_wait", "interrupted", "cancelled"}


def _now() -> float:
    return time.time()


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
                # Windows does not allow replacing a file while another thread or
                # scanner has it open without delete sharing. Browser polling can
                # briefly overlap a manifest read, so retry after the reader closes.
                time.sleep(min(0.01 * (attempt + 1), 0.10))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class OpenAIBatchManager:
    """Orchestrates multiple ordinary OpenAI interpolation jobs behind one Job ID.

    The existing OpenAIJobManager remains the authoritative worker for each temporal gap.
    This wrapper only owns the ordered gap queue and aggregates child results. That keeps
    the already-tested planner/generator/auditor/retry logic unchanged while allowing a
    selected range or the entire animation to be processed in the background.
    """

    def __init__(self, root: Path, child_manager: OpenAIJobManager):
        self.root = root
        self.child_manager = child_manager
        self.jobs_root = root / ".openai-batches"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.master_lock = threading.Lock()
        self.io_lock = threading.RLock()
        self.running: set[str] = set()
        self._mark_interrupted()

    def _job_dir(self, job_id: str) -> Path:
        if not job_id or any(ch not in "0123456789abcdef" for ch in job_id.lower()):
            raise ValueError("Invalid batch job id.")
        return self.jobs_root / job_id

    def _manifest(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _save(self, job: dict[str, Any]) -> None:
        job["updated"] = _now()
        with self.io_lock:
            _atomic_json(self._manifest(str(job["id"])), job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        try:
            path = self._manifest(job_id)
        except ValueError:
            return None
        with self.io_lock:
            if not path.is_file():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else None
            except Exception:
                return None

    def _mark_interrupted(self) -> None:
        for path in self.jobs_root.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job.get("status") in BATCH_ACTIVE_STATES:
                    job["status"] = "interrupted"
                    job["stage"] = "interrupted"
                    job["message"] = "Server restarted while this multi-gap job was running. Resume it to continue."
                    job["updated"] = _now()
                    _atomic_json(path, job)
            except Exception:
                continue

    @staticmethod
    def segment_count(request: dict[str, Any], frame_count: int) -> int:
        scope = str(request.get("scope", "pair")).lower()
        if scope == "range":
            left = max(0, min(frame_count - 1, int(request.get("left_index", 0))))
            right = max(0, min(frame_count - 1, int(request.get("right_index", frame_count - 1))))
            return max(1, right - left)
        if scope == "all":
            count = max(1, frame_count - 1)
            if str(request.get("sequence_mode", "open")).lower() == "loop" and bool(request.get("include_loop_closure")):
                count += 1
            return count
        return 1

    @staticmethod
    def _segments(request: dict[str, Any], frame_count: int) -> list[dict[str, Any]]:
        scope = str(request.get("scope", "pair")).lower()
        if scope == "range":
            left = max(0, min(frame_count - 1, int(request.get("left_index", 0))))
            right = max(0, min(frame_count - 1, int(request.get("right_index", frame_count - 1))))
            if left >= right:
                raise ValueError("For range interpolation, the first frame must come before the last frame.")
            return [
                {"left_index": index, "right_index": index + 1, "loop_closure": False}
                for index in range(left, right)
            ]
        if scope == "all":
            segments = [
                {"left_index": index, "right_index": index + 1, "loop_closure": False}
                for index in range(frame_count - 1)
            ]
            if str(request.get("sequence_mode", "open")).lower() == "loop" and bool(request.get("include_loop_closure")):
                segments.append({"left_index": frame_count - 1, "right_index": 0, "loop_closure": True})
            return segments
        raise ValueError("Batch interpolation requires scope=range or scope=all.")

    def create(self, uploads: list[tuple[str, bytes]], request: dict[str, Any]) -> dict[str, Any]:
        if len(uploads) < 2:
            raise ValueError("OpenAI interpolation needs at least two frames.")
        segments = self._segments(request, len(uploads))
        job_id = uuid.uuid4().hex
        source_dir = self._job_dir(job_id) / "source"
        source_dir.mkdir(parents=True)

        stored: list[dict[str, Any]] = []
        for index, (name, payload) in enumerate(uploads):
            path = source_dir / f"{index:06d}.bin"
            path.write_bytes(payload)
            stored.append({"name": name, "path": str(path.relative_to(self._job_dir(job_id)))})

        tasks = [
            {
                "index": index,
                **segment,
                "status": "pending",
                "child_id": None,
            }
            for index, segment in enumerate(segments)
        ]
        normal_request = dict(request)
        normal_request["scope"] = str(request.get("scope", "range")).lower()

        job = {
            "id": job_id,
            "kind": "batch",
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": f"Queued {len(tasks)} OpenAI interpolation gap{'s' if len(tasks) != 1 else ''}",
            "created": _now(),
            "updated": _now(),
            "request": normal_request,
            "sources": stored,
            "tasks": tasks,
            "current_task": None,
            "current_child_id": None,
            "error": None,
        }
        self._save(job)
        self._start(job_id)
        return self.public(job)

    def _uploads(self, job: dict[str, Any]) -> list[tuple[str, bytes]]:
        root = self._job_dir(str(job["id"]))
        return [
            (str(item.get("name", f"frame-{index + 1}.png")), (root / str(item["path"])).read_bytes())
            for index, item in enumerate(job.get("sources", []))
        ]

    def _start(self, job_id: str) -> None:
        with self.master_lock:
            if job_id in self.running:
                return
            self.running.add(job_id)
        thread = threading.Thread(target=self._work, args=(job_id,), daemon=True, name=f"openai-batch-{job_id[:8]}")
        thread.start()

    def _spent(self, job: dict[str, Any]) -> float:
        total = 0.0
        for task in job.get("tasks", []):
            child_id = task.get("child_id")
            if not child_id:
                continue
            child = self.child_manager.get(str(child_id))
            if child:
                total += float(child.get("spent_usd", 0.0))
        return total

    def _work(self, job_id: str) -> None:
        try:
            job = self.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["stage"] = "starting"
            job["error"] = None
            job["message"] = "Starting multi-gap OpenAI interpolation"
            self._save(job)

            while True:
                job = self.get(job_id)
                if not job:
                    return
                tasks = list(job.get("tasks", []))
                pending_index = next((i for i, item in enumerate(tasks) if item.get("status") != "done"), None)
                if pending_index is None:
                    job["status"] = "done"
                    job["stage"] = "done"
                    job["progress"] = 100
                    job["current_task"] = None
                    job["current_child_id"] = None
                    job["message"] = "All selected animation gaps completed"
                    self._save(job)
                    return

                task = tasks[pending_index]
                child_id = task.get("child_id")
                if not child_id:
                    spent = self._spent(job)
                    maximum = max(0.0, float(job.get("request", {}).get("max_spend_usd", 0.0) or 0.0))
                    remaining_budget = 0.0 if maximum <= 0 else maximum - spent
                    if maximum > 0 and remaining_budget <= 0:
                        job["status"] = "budget_wait"
                        job["stage"] = "budget"
                        job["current_task"] = pending_index
                        job["message"] = f"The ${maximum:.2f} batch spend limit has been reached before the next gap."
                        self._save(job)
                        return

                    child_request = dict(job.get("request", {}))
                    child_request["scope"] = "pair"
                    child_request["left_index"] = int(task["left_index"])
                    child_request["right_index"] = int(task["right_index"])
                    child_request["loop_closure"] = bool(task.get("loop_closure"))
                    child_request["max_spend_usd"] = remaining_budget if maximum > 0 else 0.0

                    child = self.child_manager.create(self._uploads(job), child_request)
                    task["child_id"] = child["id"]
                    task["status"] = "running"
                    tasks[pending_index] = task
                    job["tasks"] = tasks
                    job["current_task"] = pending_index
                    job["current_child_id"] = child["id"]
                    job["status"] = "waiting_child"
                    job["stage"] = "interpolating"
                    job["message"] = (
                        f"Processing gap {pending_index + 1}/{len(tasks)}: "
                        f"frame {int(task['left_index']) + 1} → frame {int(task['right_index']) + 1}"
                    )
                    job["progress"] = round((pending_index / max(1, len(tasks))) * 100, 1)
                    self._save(job)
                    child_id = child["id"]

                while True:
                    child = self.child_manager.get(str(child_id))
                    if not child:
                        raise RuntimeError(f"Child OpenAI job {child_id} disappeared.")
                    status = str(child.get("status", ""))
                    if status in ACTIVE_STATES:
                        job = self.get(job_id) or job
                        job["status"] = "waiting_child"
                        job["stage"] = str(child.get("stage", "interpolating"))
                        child_progress = float(child.get("progress", 0.0))
                        job["progress"] = round(((pending_index + child_progress / 100.0) / max(1, len(tasks))) * 100, 1)
                        job["message"] = (
                            f"Gap {pending_index + 1}/{len(tasks)} · {child.get('message', status)}"
                        )
                        self._save(job)
                        time.sleep(0.6)
                        continue
                    if status == "interrupted":
                        self.child_manager.resume(str(child_id))
                        time.sleep(0.2)
                        continue
                    if status == "done":
                        job = self.get(job_id) or job
                        tasks = list(job.get("tasks", []))
                        tasks[pending_index]["status"] = "done"
                        job["tasks"] = tasks
                        job["current_task"] = None
                        job["current_child_id"] = None
                        job["progress"] = round(((pending_index + 1) / max(1, len(tasks))) * 100, 1)
                        self._save(job)
                        break
                    if status in {"needs_review", "budget_wait"}:
                        job = self.get(job_id) or job
                        job["status"] = status
                        job["stage"] = str(child.get("stage", status))
                        job["current_task"] = pending_index
                        job["current_child_id"] = child_id
                        job["message"] = (
                            f"Gap {pending_index + 1}/{len(tasks)} paused: {child.get('message', status)}"
                        )
                        self._save(job)
                        return
                    if status in {"error", "cancelled"}:
                        job = self.get(job_id) or job
                        job["status"] = status
                        job["stage"] = status
                        job["current_task"] = pending_index
                        job["current_child_id"] = child_id
                        job["error"] = child.get("error") or child.get("message") or f"Child job {status}."
                        job["message"] = f"Gap {pending_index + 1}/{len(tasks)} failed"
                        self._save(job)
                        return
                    time.sleep(0.6)
        except Exception as exc:
            job = self.get(job_id)
            if job:
                job["status"] = "error"
                job["stage"] = "error"
                job["error"] = str(exc)
                job["message"] = "Multi-gap OpenAI interpolation failed"
                self._save(job)
        finally:
            with self.master_lock:
                self.running.discard(job_id)

    def resume(self, job_id: str, max_spend_usd: float | None = None) -> dict[str, Any]:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.get("status") in BATCH_ACTIVE_STATES:
            return self.public(job)
        if job.get("status") in {"done", "cancelled"}:
            return self.public(job)
        if max_spend_usd is not None:
            request = dict(job.get("request", {}))
            request["max_spend_usd"] = max(0.0, float(max_spend_usd))
            job["request"] = request
        job["status"] = "queued"
        job["stage"] = "queued"
        job["error"] = None
        job["message"] = "Multi-gap job queued to continue"
        self._save(job)
        self._start(job_id)
        return self.public(job)

    def current_child(self, job_id: str) -> str | None:
        job = self.get(job_id)
        if not job:
            return None
        child = job.get("current_child_id")
        return str(child) if child else None

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        accepted: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        estimated_next = 0.0
        pending_target = None

        for task in job.get("tasks", []):
            child_id = task.get("child_id")
            child_public = None
            if child_id:
                child = self.child_manager.get(str(child_id))
                if child:
                    child_public = self.child_manager.public(child)
                    for attempt in child_public.get("attempts", []):
                        attempts.append({
                            **attempt,
                            "job_id": child_id,
                            "segment_index": task.get("index"),
                            "left_index": task.get("left_index"),
                            "right_index": task.get("right_index"),
                            "loop_closure": bool(task.get("loop_closure")),
                        })
                    for result in child_public.get("accepted", []):
                        accepted.append({
                            **result,
                            "job_id": child_id,
                            "segment_index": task.get("index"),
                            "left_index": task.get("left_index"),
                            "right_index": task.get("right_index"),
                            "loop_closure": bool(task.get("loop_closure")),
                        })
                    if str(child_id) == str(job.get("current_child_id") or ""):
                        estimated_next = float(child_public.get("estimated_next_usd", 0.0) or 0.0)
                        pending_target = child_public.get("pending_target")
            segments.append({
                "index": task.get("index"),
                "left_index": task.get("left_index"),
                "right_index": task.get("right_index"),
                "loop_closure": bool(task.get("loop_closure")),
                "status": child_public.get("status") if child_public else task.get("status", "pending"),
                "child_id": child_id,
            })

        attempts.sort(key=lambda item: (int(item.get("segment_index") or 0), float(item.get("target_fraction") or 0), int(item.get("attempt") or 0)))
        accepted.sort(key=lambda item: (int(item.get("segment_index") or 0), float(item.get("target_fraction") or 0)))
        return {
            "id": job.get("id"),
            "kind": "batch",
            "status": job.get("status"),
            "stage": job.get("stage"),
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "error": job.get("error"),
            "created": job.get("created"),
            "updated": job.get("updated"),
            "spent_usd": self._spent(job),
            "estimated_next_usd": estimated_next,
            "request": job.get("request", {}),
            "attempts": attempts,
            "accepted": accepted,
            "pending_target": pending_target,
            "segments": segments,
            "current_child_id": job.get("current_child_id"),
        }

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if not job:
            return
        if job.get("status") in BATCH_ACTIVE_STATES:
            raise RuntimeError("Cannot delete a running multi-gap job.")
        for task in job.get("tasks", []):
            child_id = task.get("child_id")
            if not child_id:
                continue
            try:
                self.child_manager.delete(str(child_id))
            except Exception:
                pass
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

    def cleanup(self) -> None:
        now = _now()
        for path in self.jobs_root.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                status = str(job.get("status", ""))
                if status in BATCH_ACTIVE_STATES:
                    continue
                age = now - float(job.get("updated", job.get("created", now)))
                ttl = 7 * 86400 if status in {"needs_review", "budget_wait", "interrupted"} else 86400
                if age > ttl:
                    shutil.rmtree(path.parent, ignore_errors=True)
            except Exception:
                continue