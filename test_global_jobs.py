from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from global_jobs import GlobalJobStore


class GlobalJobStoreTests(unittest.TestCase):
    def test_sources_and_result_survive_store_recreation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = GlobalJobStore(root)
            job_id = "a" * 32
            _, paths = store.begin_render(
                job_id,
                [("one.png", b"one"), ("two.webp", b"two")],
                {"duration": 100, "rife_multiplier": 1},
            )
            self.assertEqual([path.read_bytes() for path in paths], [b"one", b"two"])
            output = store.job_dir(job_id) / "animation.webp"
            output.write_bytes(b"finished-webp")
            store.update(job_id, status="done", progress=100, message="done")

            restored = GlobalJobStore(root)
            job = restored.get(job_id)
            self.assertIsNotNone(job)
            public = restored.public(job)
            self.assertEqual(public["status"], "done")
            self.assertEqual(public["source_count"], 2)
            self.assertTrue(public["output_available"])
            self.assertEqual(restored.output_path(job_id).read_bytes(), b"finished-webp")

    def test_active_render_becomes_interrupted_after_server_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = GlobalJobStore(root)
            job_id = "b" * 32
            store.begin_render(job_id, [("one.png", b"one")], {"duration": 100})
            store.update(job_id, status="running", progress=55)

            restored = GlobalJobStore(root)
            job = restored.get(job_id)
            self.assertEqual(job["status"], "interrupted")
            self.assertIn("Server restarted", job["message"])

    def test_new_job_does_not_delete_previous_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = GlobalJobStore(root)
            first = "c" * 32
            second = "d" * 32
            store.ensure(first)
            store.ensure(second)
            self.assertIsNotNone(store.get(first))
            self.assertIsNotNone(store.get(second))

    def test_openai_children_are_linked_to_global_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = GlobalJobStore(Path(temporary))
            job_id = "e" * 32
            child = "f" * 32
            store.ensure(job_id)
            store.attach_openai(job_id, child)
            store.attach_openai(job_id, child)
            public = store.public(store.get(job_id))
            self.assertEqual(public["openai_jobs"], [child])
            self.assertEqual(public["last_openai_job_id"], child)


if __name__ == "__main__":
    unittest.main()
