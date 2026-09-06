from __future__ import annotations

import threading
from typing import Any

from openai_job_features import EnhancedOpenAIJobManager
from resilient_bridge import OpenAIJobCancelled


class CancellableEnhancedOpenAIJobManager(EnhancedOpenAIJobManager):
    """Enhanced OpenAI manager with durable user cancellation.

    Cancellation is cooperative between paid stages. If the user stops a job while an
    HTTP request is already in flight, that single request may still finish, but the
    bridge raises before the next planner/generator/auditor call and the worker cannot
    overwrite the persisted cancelled state.
    """

    def __init__(self, root):
        self._cancel_lock = threading.Lock()
        self._cancelled_ids: set[str] = set()
        super().__init__(root)

    def is_cancelled(self, job_id: str) -> bool:
        with self._cancel_lock:
            if job_id in self._cancelled_ids:
                return True
        job = self.get(job_id)
        return bool(job and (job.get("cancel_requested") or job.get("status") == "cancelled"))

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._cancel_lock:
            self._cancelled_ids.add(job_id)
        with self._lock_for(job_id):
            job = self.get(job_id)
            if not job:
                raise KeyError(job_id)
            job["cancel_requested"] = True
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["estimated_next_usd"] = 0.0
            job["error"] = None
            job["message"] = (
                "Stopped by user. No further OpenAI stages or automatic retries will be started. "
                "A request that was already in flight when Stop was pressed may still finish remotely."
            )
            self._save(job)
            return self.public(job)

    def _update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        current = self.get(job_id)
        if current and (current.get("cancel_requested") or current.get("status") == "cancelled"):
            # Once cancelled, background worker cleanup/error paths are never allowed
            # to resurrect the job as running/replanning/error/needs_review.
            if changes.get("status") != "cancelled":
                return current
        return super()._update(job_id, **changes)

    def _attempt(self, job_id: str, target: float, preplan=None) -> bool:
        if self.is_cancelled(job_id):
            raise OpenAIJobCancelled(f"OpenAI job {job_id} was stopped by the user.")
        setter = getattr(self.bridge, "set_current_job", None)
        clearer = getattr(self.bridge, "clear_current_job", None)
        if callable(setter):
            setter(job_id, self.is_cancelled)
        try:
            result = super()._attempt(job_id, target, preplan=preplan)
            if self.is_cancelled(job_id):
                raise OpenAIJobCancelled(f"OpenAI job {job_id} was stopped by the user.")
            return result
        finally:
            if callable(clearer):
                clearer()
