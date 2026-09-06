from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from openai_interrogator import ACTIVE_STATES, BridgeClient, OpenAIJobManager, _fraction_key, make_contact_sheet


AUTO_BUDGET_HARD_ATTEMPT_LIMIT = 32
MAX_AUDITOR_OVERRIDES = 32


def _atomic_manifest(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist a job manifest without losing Windows polling races."""
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


def _override_key(value: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("type", "")).strip().casefold(),
        str(value.get("region", "")).strip().casefold(),
        str(value.get("description", "")).strip().casefold(),
    )


def _merge_overrides(existing: list[Any], learned: list[Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in [*existing, *learned]:
        if not isinstance(raw, dict):
            continue
        item = {
            "type": str(raw.get("type", "user_accepted_issue")).strip() or "user_accepted_issue",
            "region": str(raw.get("region", "whole frame")).strip() or "whole frame",
            "description": str(raw.get("description", "")).strip(),
            "severity": max(0, min(100, int(raw.get("severity", 0) or 0))),
        }
        if raw.get("learned_from_target") is not None:
            item["learned_from_target"] = float(raw["learned_from_target"])
        if raw.get("learned_from_attempt") is not None:
            item["learned_from_attempt"] = int(raw["learned_from_attempt"])
        if not item["description"]:
            continue
        key = _override_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged[-MAX_AUDITOR_OVERRIDES:]


class AuditorAwareBridgeClient(BridgeClient):
    """Inject job-local user audit preferences only into auditor calls.

    The generator and planner remain unchanged. A manual accept therefore teaches the
    critic what the user considers acceptable without relaxing unrelated checks.
    Thread-local state keeps concurrent background jobs isolated from one another.
    """

    def __init__(self, root: Path, port: int = 18744):
        super().__init__(root, port)
        self._audit_context = threading.local()

    def set_auditor_overrides(self, overrides: list[dict[str, Any]] | None) -> None:
        self._audit_context.overrides = list(overrides or [])

    def request(self, path: str, payload: dict[str, Any] | None, **kwargs):
        if path == "/audit" and payload is not None:
            overrides = list(getattr(self._audit_context, "overrides", []) or [])
            if overrides:
                payload = dict(payload)
                original = str(payload.get("user_instruction", "")).strip()
                learned_text = json.dumps(overrides, ensure_ascii=False, indent=2)
                preference = (
                    "AUDITOR JOB-LOCAL USER PREFERENCE OVERRIDE:\n"
                    "The user explicitly accepted earlier generated frame(s) despite the audit findings below. "
                    "For this animation job, if the same or substantially equivalent visual issue appears again, "
                    "treat that specific issue as allowed: do not list it as a violation, do not lower rubric scores "
                    "because of it, and do not reject a candidate solely because of it. Continue to evaluate every "
                    "unrelated temporal, structural, semantic, consistency, or instruction error normally. Do not "
                    "generalise these exceptions beyond the specific findings described.\n"
                    f"Accepted audit exceptions:\n{learned_text}"
                )
                payload["user_instruction"] = f"{original}\n\n{preference}" if original else preference
        return super().request(path, payload, **kwargs)


class EnhancedOpenAIJobManager(OpenAIJobManager):
    """Runtime OpenAI worker used by the web app.

    Adds Windows-safe manifests, complete sequence context, user overrides that teach
    subsequent audits, and a budget-driven automatic retry policy. The underlying
    planner/generator/auditor implementation remains in OpenAIJobManager.
    """

    def __init__(self, root: Path):
        self._manifest_io_lock = threading.RLock()
        super().__init__(root)
        self.bridge = AuditorAwareBridgeClient(root)

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

    def _normalise_request(self, request: dict[str, Any], frame_count: int) -> dict[str, Any]:
        normal = super()._normalise_request(request, frame_count)
        retry_policy = str(request.get("retry_policy", "manual")).lower().strip()
        if retry_policy not in {"manual", "auto_budget"}:
            retry_policy = "manual"
        if retry_policy == "auto_budget" and float(normal.get("max_spend_usd", 0.0)) <= 0:
            raise ValueError("Auto-retry-until-budget requires a maximum OpenAI spend greater than $0.")
        normal["retry_policy"] = retry_policy
        normal["auditor_overrides"] = _merge_overrides([], list(request.get("auditor_overrides", []) or []))
        return normal

    def estimate(self, request: dict[str, Any], frame_count: int, width: int, height: int) -> dict[str, Any]:
        normal = self._normalise_request(request, frame_count)
        estimate = super().estimate(request, frame_count, width, height)
        if normal.get("retry_policy") == "auto_budget":
            one = max(0.000001, float(estimate.get("one_attempt", 0.0) or 0.0))
            budget = float(normal["max_spend_usd"])
            estimate["budget_driven"] = True
            estimate["maximum"] = budget
            estimate["attempts_per_frame"] = "budget"
            estimate["budget_estimated_attempts"] = max(1, min(AUTO_BUDGET_HARD_ATTEMPT_LIMIT, int(budget // one)))
        return estimate

    def public(self, job: dict[str, Any]) -> dict[str, Any]:
        value = super().public(job)
        internal_attempts = list(job.get("attempts", []))
        for index, attempt in enumerate(value.get("attempts", [])):
            if index < len(internal_attempts):
                attempt["accepted_by_user"] = bool(internal_attempts[index].get("accepted_by_user"))
        accepted_internal = {
            _fraction_key(float(item.get("target_fraction", -1))): item
            for item in job.get("accepted", [])
        }
        for accepted in value.get("accepted", []):
            internal = accepted_internal.get(_fraction_key(float(accepted.get("target_fraction", -1)))) or {}
            accepted["accepted_by_user"] = bool(internal.get("accepted_by_user"))
        value["source_count"] = len(job.get("sources", []))
        value["source_names"] = [str(item.get("name", f"frame-{index + 1}.png")) for index, item in enumerate(job.get("sources", []))]
        value["auditor_overrides"] = list(job.get("request", {}).get("auditor_overrides", []) or [])
        return value

    def _attempt(self, job_id: str, target: float, preplan: dict[str, Any] | None = None) -> bool:
        job = self.get(job_id)
        overrides = list((job or {}).get("request", {}).get("auditor_overrides", []) or [])
        setter = getattr(self.bridge, "set_auditor_overrides", None)
        if callable(setter):
            setter(overrides)
        try:
            return super()._attempt(job_id, target, preplan=preplan)
        finally:
            if callable(setter):
                setter([])

    def _propagate_auditor_overrides_to_parent_batches(
        self,
        child_job_id: str,
        overrides: list[dict[str, Any]],
    ) -> None:
        """Carry learned audit preferences to later gaps in the same batch animation."""
        if not overrides:
            return
        batches_root = self.root / ".openai-batches"
        if not batches_root.is_dir():
            return
        for manifest in batches_root.glob("*/job.json"):
            try:
                batch = json.loads(manifest.read_text(encoding="utf-8"))
                tasks = list(batch.get("tasks", []))
                belongs = str(batch.get("current_child_id") or "") == child_job_id or any(
                    str(task.get("child_id") or "") == child_job_id for task in tasks
                )
                if not belongs:
                    continue
                request = dict(batch.get("request", {}))
                request["auditor_overrides"] = _merge_overrides(
                    list(request.get("auditor_overrides", []) or []),
                    overrides,
                )
                batch["request"] = request
                batch["updated"] = time.time()
                _atomic_manifest(manifest, batch)
            except Exception:
                continue

    def accept_latest(self, job_id: str, target_fraction: float | None = None) -> dict[str, Any]:
        """Accept the latest generated candidate, learn its audit exceptions, and continue."""
        with self._lock_for(job_id):
            job = self.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.get("status") in ACTIVE_STATES:
                raise ValueError("Wait for the current OpenAI attempt to finish before overriding the auditor.")
            if job.get("status") == "cancelled":
                raise ValueError("A cancelled job cannot be continued.")

            attempts = list(job.get("attempts", []))
            if target_fraction is None:
                if job.get("pending_target") is not None:
                    target_fraction = float(job["pending_target"])
                elif attempts:
                    target_fraction = float(attempts[-1].get("target_fraction", 0.5))
                else:
                    raise ValueError("This job has no generated attempt to accept.")

            key = _fraction_key(float(target_fraction))
            matching = [
                item for item in attempts
                if _fraction_key(float(item.get("target_fraction", -1))) == key
                and str(item.get("candidate_path", ""))
            ]
            if not matching:
                raise ValueError("No generated image exists for that interpolation target.")
            latest = matching[-1]
            candidate_path = self._job_dir(job_id) / str(latest["candidate_path"])
            if not candidate_path.is_file():
                raise ValueError("The generated candidate image is no longer available.")

            accepted_dir = self._job_dir(job_id) / "accepted"
            accepted_dir.mkdir(parents=True, exist_ok=True)
            accepted_path = accepted_dir / f"t-{key.replace('.', '_')}.png"
            shutil.copyfile(candidate_path, accepted_path)

            latest["accepted"] = True
            latest["accepted_by_user"] = True
            latest["user_override_reason"] = "Auditor rejection ignored by the user and learned as a job-local audit preference."

            audit = latest.get("audit") if isinstance(latest.get("audit"), dict) else {}
            learned: list[dict[str, Any]] = []
            for raw in list(audit.get("violations", []) or []):
                if not isinstance(raw, dict):
                    continue
                learned.append({
                    "type": str(raw.get("type", "user_accepted_issue")),
                    "region": str(raw.get("region", "whole frame")),
                    "description": str(raw.get("description", "")),
                    "severity": int(raw.get("severity", 0) or 0),
                    "learned_from_target": float(target_fraction),
                    "learned_from_attempt": int(latest.get("attempt", 0) or 0),
                })
            if not learned:
                fallback = str(audit.get("corrective_instruction", "")).strip()
                if fallback:
                    learned.append({
                        "type": "user_accepted_audit_rejection",
                        "region": "whole frame",
                        "description": fallback,
                        "severity": 0,
                        "learned_from_target": float(target_fraction),
                        "learned_from_attempt": int(latest.get("attempt", 0) or 0),
                    })

            request = dict(job.get("request", {}))
            request["auditor_overrides"] = _merge_overrides(
                list(request.get("auditor_overrides", []) or []),
                learned,
            )
            job["request"] = request

            job["accepted"] = [
                item for item in job.get("accepted", [])
                if _fraction_key(float(item.get("target_fraction", -1))) != key
            ]
            job["accepted"].append({
                "target_fraction": float(target_fraction),
                "path": str(accepted_path.relative_to(self._job_dir(job_id))),
                "accepted_by_user": True,
            })
            job["targets_remaining"] = [
                float(value) for value in job.get("targets_remaining", [])
                if _fraction_key(float(value)) != key
            ]
            job["pending_target"] = None
            job["status"] = "queued"
            job["stage"] = "queued"
            job["error"] = None
            job["estimated_next_usd"] = 0.0
            learned_count = len(request["auditor_overrides"])
            job["message"] = (
                f"User accepted the generated frame at t={float(target_fraction):.3f}; "
                f"the auditor will treat {learned_count} learned issue{'s' if learned_count != 1 else ''} as allowed for the rest of this job."
            )
            job.get("request", {}).pop("user_feedback", None)
            self._save(job)
            self._propagate_auditor_overrides_to_parent_batches(job_id, request["auditor_overrides"])

        self._start(job_id)
        return self.public(self.get(job_id) or job)

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
                existing = sum(
                    1 for item in job.get("attempts", [])
                    if _fraction_key(float(item.get("target_fraction", -1))) == _fraction_key(target)
                )
                auto_budget = str(request.get("retry_policy", "manual")) == "auto_budget"
                if auto_budget:
                    attempts_left = max(0, AUTO_BUDGET_HARD_ATTEMPT_LIMIT - existing)
                else:
                    max_attempts = 1 + int(request.get("auto_retries", 0))
                    attempts_left = max(1, max_attempts - existing)

                if auto_budget and attempts_left == 0:
                    self._update(
                        job_id,
                        status="needs_review",
                        stage="review",
                        pending_target=target,
                        message=f"Automatic retry safety limit ({AUTO_BUDGET_HARD_ATTEMPT_LIMIT} attempts for one frame) reached. Review the latest candidate before continuing.",
                    )
                    return

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
                    if auto_budget:
                        attempts_now = sum(
                            1 for item in current.get("attempts", [])
                            if _fraction_key(float(item.get("target_fraction", -1))) == _fraction_key(target)
                        )
                        self._update(
                            job_id,
                            status="replanning",
                            stage="replanning",
                            pending_target=target,
                            message=f"Auditor rejected attempt {attempts_now}; automatically learning from it and retrying while the configured budget allows.",
                        )

                job = self.get(job_id)
                if not job:
                    return
                if not accepted:
                    self._update(
                        job_id,
                        status="needs_review",
                        stage="review",
                        pending_target=target,
                        message=(
                            "Automatic retry limit reached before acceptance. You can accept the latest frame anyway or add feedback and continue."
                            if auto_budget
                            else "The auditor rejected the latest frame. Accept it anyway, add feedback and try again, or use automatic budget-driven retries."
                        ),
                    )
                    return

                job["targets_remaining"] = [
                    float(value) for value in job.get("targets_remaining", [])
                    if _fraction_key(float(value)) != _fraction_key(target)
                ]
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
