from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from cancellable_openai import CancellableEnhancedOpenAIJobManager


class CancellableOpenAITests(unittest.TestCase):
    def test_cancel_does_not_wait_for_worker_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = CancellableEnhancedOpenAIJobManager(Path(temporary))
            job_id = "a" * 32
            manager._job_dir(job_id).mkdir(parents=True, exist_ok=True)
            manager._save({
                "id": job_id,
                "status": "generating",
                "stage": "generating",
                "progress": 10,
                "message": "in flight",
                "error": None,
                "created": time.time(),
                "updated": time.time(),
                "spent_usd": 0.0,
                "estimated_next_usd": 0.1,
                "request": {},
                "attempts": [],
                "accepted": [],
                "pending_target": 0.5,
                "sources": [],
            })

            worker_lock = manager._lock_for(job_id)
            worker_lock.acquire()
            finished = threading.Event()
            error = []

            def stop():
                try:
                    manager.cancel(job_id)
                except Exception as exc:  # pragma: no cover - assertion reports it
                    error.append(exc)
                finally:
                    finished.set()

            thread = threading.Thread(target=stop, daemon=True)
            thread.start()
            try:
                self.assertTrue(finished.wait(0.5), "Stop blocked behind the long-lived worker lock")
            finally:
                worker_lock.release()
                thread.join(timeout=1)

            self.assertEqual(error, [])
            job = manager.get(job_id)
            self.assertIsNotNone(job)
            self.assertEqual(job["status"], "cancelled")
            self.assertTrue(job["cancel_requested"])


if __name__ == "__main__":
    unittest.main()
